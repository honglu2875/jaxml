# coding=utf-8
# Copyright 2024 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DeepSeek-V3 / R1 building blocks (MLA attention + fine-grained MoE). The
# routing and MLA math mirror transformers' modeling_deepseek_v3 so the modules
# can be checked for forward-parity against the reference implementation.

import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.partitioning import with_sharding_constraint

from ..cache import KVCache
from ..nn.attention import Attention, _validate_cos_sin, _validate_hidden_states
from ..nn.embedding import Embed
from ..nn.linear import DenseGeneral
from ..nn.module import Block
from ..nn.norms import RMSNorm
from ..outputs import AttentionOutput, BaseModelOutputWithCache, CausalLMOutputWithCache, DecoderOutput
from ._utils import (
    normalize_model_output_flags,
    prepare_default_attention_mask,
    prepare_model_inputs,
    prepare_position_ids,
    slice_last_n_logits_hidden_states,
)
from .llama import LlamaMLP

_GMM_TILE = 128  # megablox gmm tiles the token (M) dimension by this; pad M to a multiple.


class DeepseekMoE(Block):
    """Fine-grained mixture-of-experts FFN with shared experts (DeepSeek-V3).

    Routed experts are stored as stacked 3D kernels (matching the reference
    `DeepseekV3NaiveMoe` layout). Two evaluation paths share the same parameters:

    - ``dense`` (default off TPU / reference): every expert is evaluated for every
      token and weighted by a dense ``(tokens, experts)`` routing matrix that is
      zero for non-selected experts. Parity-exact and portable, but costs
      ``n_routed_experts`` x the active compute.
    - ``gmm`` (TPU): a sparse path that dispatches tokens to their selected experts
      (sort + gather), runs the expert FFN with the Pallas grouped-matmul kernel
      ``megablox.gmm``, and combines via a weighted scatter-add. Equivalent to the
      dense path up to TPU bf16, and avoids the ``n_routed_experts`` x blowup.

    ``moe_implementation`` selects the path: ``"auto"`` uses gmm on TPU and
    dense otherwise.
    """

    kernel_init: Any = nn.initializers.normal(stddev=0.02)
    moe_implementation: str = "auto"  # "auto" | "dense" | "gmm"

    def setup(self):
        cfg = self.config
        if not cfg.is_moe:
            raise ValueError("DeepseekMoE requires a config with n_routed_experts set.")
        self.n_routed_experts = cfg.n_routed_experts
        self.moe_intermediate_size = cfg.moe_intermediate_size
        self.top_k = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.norm_topk_prob = cfg.norm_topk_prob
        self.routed_scaling_factor = cfg.routed_scaling_factor

        H = self.hidden_size
        inter = self.moe_intermediate_size
        E = self.n_routed_experts

        # Router: produces one logit per expert. Computed in float32 (see __call__).
        self.gate_kernel = self.param(
            "gate_kernel",
            nn.with_logical_partitioning(self.kernel_init, ("embed", "expert")),
            (H, E),
            jnp.float32,
        )
        self.e_score_correction_bias = self.param(
            "e_score_correction_bias",
            nn.with_logical_partitioning(nn.initializers.zeros, ("expert",)),
            (E,),
            jnp.float32,
        )

        # Stacked routed-expert weights. gate_up is fused (gate first, then up).
        self.gate_up_kernel = self.param(
            "gate_up_kernel",
            nn.with_logical_partitioning(self.kernel_init, ("expert", "embed", "intermediate")),
            (E, H, 2 * inter),
            self.dtype,
        )
        self.down_kernel = self.param(
            "down_kernel",
            nn.with_logical_partitioning(self.kernel_init, ("expert", "intermediate", "embed")),
            (E, inter, H),
            self.dtype,
        )

        # Shared expert: a single SwiGLU MLP applied to every token.
        shared_intermediate = self.moe_intermediate_size * cfg.n_shared_experts
        self.shared_gate_proj = DenseGeneral(
            features=shared_intermediate,
            dtype=self.dtype,
            kernel_axes=("embed", "intermediate"),
            name="shared_gate_proj",
        )
        self.shared_up_proj = DenseGeneral(
            features=shared_intermediate,
            dtype=self.dtype,
            kernel_axes=("embed", "intermediate"),
            name="shared_up_proj",
        )
        self.shared_down_proj = DenseGeneral(
            features=H,
            dtype=self.dtype,
            kernel_axes=("intermediate", "embed"),
            name="shared_down_proj",
        )

    def _route_topk(self, hidden_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(topk_idx (T, top_k) int32, topk_weights (T, top_k) float32)``.

        Mirrors `DeepseekV3MoE.route_tokens_to_experts`: sigmoid scores, additive
        correction bias for *selection only*, group-limited top-k, then top-k
        expert selection with optional sum-normalization and scaling.
        """
        E = self.n_routed_experts
        G = self.n_group
        per_group = E // G
        T = hidden_flat.shape[0]

        logits = jnp.einsum("th,he->te", hidden_flat.astype(jnp.float32), self.gate_kernel.astype(jnp.float32))
        scores = jax.nn.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(jnp.float32)

        # Group-limited routing: keep only the topk_group highest-scoring groups.
        grouped = choice.reshape(T, G, per_group)
        group_scores = jax.lax.top_k(grouped, 2)[0].sum(axis=-1)  # (T, G)
        _, group_idx = jax.lax.top_k(group_scores, self.topk_group)  # (T, topk_group)
        rows = jnp.arange(T)[:, None]
        group_mask = jnp.zeros((T, G), dtype=bool).at[rows, group_idx].set(True)
        score_mask = jnp.broadcast_to(group_mask[:, :, None], (T, G, per_group)).reshape(T, E)
        masked_choice = jnp.where(score_mask, choice, -jnp.inf)

        _, topk_idx = jax.lax.top_k(masked_choice, self.top_k)  # (T, top_k)
        topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)  # gather from pre-bias scores
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(axis=-1, keepdims=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_idx.astype(jnp.int32), topk_weights.astype(jnp.float32)

    def _resolve_implementation(self, seq_len: int) -> str:
        impl = self.moe_implementation
        if impl not in ("auto", "dense", "gmm"):
            raise ValueError(f"moe_implementation must be 'auto', 'dense', or 'gmm', got {impl!r}.")
        if impl != "auto":
            return impl
        # The sparse kernel avoids evaluating every routed expert during both
        # prefill and single-token decode, and is only available on TPU.
        if jax.default_backend() == "tpu":
            return "gmm"
        return "dense"

    def _moe_dense(self, hidden_flat: jnp.ndarray, topk_idx: jnp.ndarray, topk_weights: jnp.ndarray) -> jnp.ndarray:
        """Dense reference: evaluate all experts, weight by a sparse routing matrix."""
        T, H = hidden_flat.shape
        inter = self.moe_intermediate_size
        rows = jnp.arange(T)[:, None]
        routing = jnp.zeros((T, self.n_routed_experts), dtype=jnp.float32).at[rows, topk_idx].set(topk_weights)

        xd = hidden_flat.astype(self.dtype)
        gate_up = jnp.einsum("th,ehf->tef", xd, self.gate_up_kernel)  # (T, E, 2I)
        gate, up = gate_up[..., :inter], gate_up[..., inter:]
        expert_hidden = jax.nn.silu(gate) * up
        expert_out = jnp.einsum("tei,eih->teh", expert_hidden, self.down_kernel)  # (T, E, H)
        return jnp.einsum("te,teh->th", routing.astype(self.dtype), expert_out)  # (T, H)

    def _moe_gmm(self, hidden_flat: jnp.ndarray, topk_idx: jnp.ndarray, topk_weights: jnp.ndarray) -> jnp.ndarray:
        """Sparse path: sort-dispatch -> megablox grouped GEMM -> weighted scatter-combine."""
        from jax.experimental.pallas.ops.tpu import megablox

        T, H = hidden_flat.shape
        E, k, inter = self.n_routed_experts, self.top_k, self.moe_intermediate_size
        M = T * k
        m_pad = ((M + _GMM_TILE - 1) // _GMM_TILE) * _GMM_TILE

        flat_expert = topk_idx.reshape(-1)  # (M,)
        flat_weight = topk_weights.reshape(-1)  # (M,)
        token_of = jnp.repeat(jnp.arange(T, dtype=jnp.int32), k)  # original token per assignment

        # Dispatch: sort assignments by expert so each expert's tokens are contiguous.
        order = jnp.argsort(flat_expert)
        sorted_token = token_of[order]
        sorted_weight = flat_weight[order]
        group_sizes = jnp.bincount(flat_expert, length=E).astype(jnp.int32)  # sums to M

        # Pad M up to a tile multiple; padding rows are zeros appended after the last
        # expert's block, so absorb them into the final group (their output is discarded).
        lhs = hidden_flat[sorted_token].astype(self.dtype)
        lhs = jnp.concatenate([lhs, jnp.zeros((m_pad - M, H), lhs.dtype)], axis=0)
        group_sizes = group_sizes.at[E - 1].add(m_pad - M)

        # Expert FFN via the Pallas grouped matmul.
        gate_up = megablox.gmm(lhs, self.gate_up_kernel, group_sizes)  # (m_pad, 2I)
        expert_hidden = jax.nn.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        expert_out = megablox.gmm(expert_hidden, self.down_kernel, group_sizes)  # (m_pad, H)

        # Combine: scatter weighted expert outputs back to their original tokens.
        contrib = expert_out[:M] * sorted_weight[:, None].astype(expert_out.dtype)
        return jnp.zeros((T, H), expert_out.dtype).at[sorted_token].add(contrib)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        bsz, seq_len, H = x.shape
        hidden_flat = x.reshape(-1, H)

        topk_idx, topk_weights = self._route_topk(hidden_flat)
        if self._resolve_implementation(seq_len) == "gmm":
            routed = self._moe_gmm(hidden_flat, topk_idx, topk_weights)
        else:
            routed = self._moe_dense(hidden_flat, topk_idx, topk_weights)
        routed = routed.reshape(bsz, seq_len, H)

        shared = self.shared_down_proj(jax.nn.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x))
        return routed + shared


def _apply_rope_interleave(q_rot, k_rot, cos, sin):
    """DeepSeek interleaved RoPE applied to the decoupled rope slices.

    Mirrors transformers' ``apply_rotary_pos_emb_interleave``: ``cos``/``sin`` are
    ``cat(freqs, freqs)`` so only the first half holds the per-pair angle; the
    even/odd input lanes are rotated and re-concatenated. Inputs use the
    ``(batch, seq, heads, rope)`` layout (cos/sin are ``(batch, seq, rope)``).
    """
    half = cos.shape[-1] // 2
    cos_h = cos[..., :half][:, :, None, :]
    sin_h = sin[..., :half][:, :, None, :]

    def _rotate(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return jnp.concatenate([x1 * cos_h - x2 * sin_h, x2 * cos_h + x1 * sin_h], axis=-1)

    return _rotate(q_rot), _rotate(k_rot)


class MLAAttention(Attention):
    """Multi-head Latent Attention (DeepSeek-V3).

    Query and key/value are produced through low-rank latent projections; RoPE is
    applied to a small decoupled slice only. The key/value heads have different
    dimensions (``qk_head_dim`` vs ``v_head_dim``), so the value is zero-padded to
    ``qk_head_dim`` — exactly as the reference does for its flash path — which lets
    this class reuse the standard ``mha`` core and ``KVCache`` (which require
    matching key/value shapes). The attention output is sliced back to
    ``v_head_dim`` before the output projection.

    ``cos_sin`` are the per-token (already position-indexed) cosine/sine arrays of
    shape ``(batch, seq, qk_rope_head_dim)``, matching ``AttentionWithRoPE``'s
    convention of receiving the rotary tables from the caller.
    """

    def setup(self):
        cfg = self.config
        if not cfg.is_mla:
            raise ValueError("MLAAttention requires a config with kv_lora_rank set.")
        nh = cfg.num_heads
        self.num_key_value_heads = nh  # MLA materializes one k/v per query head.
        nope, rope, v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        qk = nope + rope

        if cfg.q_lora_rank is None:
            self.q_proj = DenseGeneral(
                features=(nh, qk),
                axis=-1,
                kernel_axes=("embed", "heads", "head_states"),
                dtype=self.dtype,
                use_bias=cfg.use_bias,
                name="q_proj",
            )
        else:
            self.q_a_proj = DenseGeneral(
                features=cfg.q_lora_rank,
                axis=-1,
                kernel_axes=("embed", "q_lora"),
                dtype=self.dtype,
                use_bias=cfg.use_bias,
                name="q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, eps=cfg.norm_eps, dtype=self.dtype, upcast=True)
            self.q_b_proj = DenseGeneral(
                features=(nh, qk),
                axis=-1,
                kernel_axes=("q_lora", "heads", "head_states"),
                dtype=self.dtype,
                name="q_b_proj",
            )

        self.kv_a_proj_with_mqa = DenseGeneral(
            features=cfg.kv_lora_rank + rope,
            axis=-1,
            kernel_axes=("embed", "kv_lora_rope"),
            dtype=self.dtype,
            use_bias=cfg.use_bias,
            name="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, eps=cfg.norm_eps, dtype=self.dtype, upcast=True)
        self.kv_b_proj = DenseGeneral(
            features=(nh, nope + v),
            axis=-1,
            kernel_axes=("kv_lora", "heads", "head_states"),
            dtype=self.dtype,
            name="kv_b_proj",
        )
        self.o_proj = DenseGeneral(
            features=cfg.hidden_size,
            axis=-1,
            kernel_axes=("heads_merged", "embed"),
            dtype=self.dtype,
            use_bias=cfg.use_bias,
            name="o_proj",
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos_sin: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_flash: bool = False,
        sliding_window: int | None = None,
    ) -> AttentionOutput:
        hidden_states = _validate_hidden_states(hidden_states)
        cfg = self.config
        if cfg.mla_use_nope:
            if cos_sin is not None:
                raise ValueError("cos_sin must be None when mla_use_nope=True.")
            cos = sin = None
        else:
            cos, sin = _validate_cos_sin(cos_sin)
        bsz, seq_len, _ = hidden_states.shape
        nh = cfg.num_heads
        nope, rope, v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        qk = nope + rope

        # Query: optional low-rank compression, then split into nope / rope parts.
        if cfg.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape(bsz, seq_len, nh, qk)
        q_nope, q_rot = q[..., :nope], q[..., nope:]

        # Key/value: shared latent (kv_lora_rank) + a single-head decoupled rope key.
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = compressed[..., : cfg.kv_lora_rank], compressed[..., cfg.kv_lora_rank :]
        kv = self.kv_b_proj(self.kv_a_layernorm(k_pass)).reshape(bsz, seq_len, nh, nope + v)
        k_nope, value = kv[..., :nope], kv[..., nope:]
        k_rot = k_rot.reshape(bsz, seq_len, 1, rope)

        if not cfg.mla_use_nope:
            q_rot, k_rot = _apply_rope_interleave(q_rot, k_rot, cos, sin)
        k_rot = jnp.broadcast_to(k_rot, (bsz, seq_len, nh, rope))

        query_states = jnp.concatenate([q_nope, q_rot], axis=-1)  # (b, s, nh, qk_head_dim)
        key_states = jnp.concatenate([k_nope, k_rot], axis=-1)  # (b, s, nh, qk_head_dim)
        # Pad value to qk_head_dim so key/value shapes match (reference does the same
        # for flash); the padded lanes are sliced off after attention.
        if qk > v:
            value_states = jnp.concatenate([value, jnp.zeros((bsz, seq_len, nh, qk - v), dtype=value.dtype)], axis=-1)
        else:
            value_states = value

        if attention_mask is None:
            attention_mask = self._get_default_mask(hidden_states, kv_cache)
        key_states, value_states, attention_mask, kv_cache = self.apply_kv_cache(
            key_states, value_states, attention_mask, kv_cache
        )

        attn_output, attn_weight = self.mha(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            causal=True,
            sliding_window=None,
            alibi_slope=None,
            softmax_fp32=True,
            output_attentions=output_attentions,
        )

        attn_output = attn_output[..., :v].reshape(bsz, seq_len, nh * v)
        return AttentionOutput(
            attention_output=self.o_proj(attn_output),
            attention_weight=attn_weight,
            kv_cache=kv_cache,
        )


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _deepseek_yarn_inv_freq(config) -> tuple[jnp.ndarray, float]:
    """Return YaRN inverse frequencies and cosine/sine scaling.

    This mirrors Transformers' ``_compute_yarn_parameters`` for the decoupled
    MLA rotary width (``qk_rope_head_dim``), including Kimi-K2's mscale fields.
    """
    dim = config.qk_rope_head_dim
    base = config.rope_theta
    factor = config.rope_factor
    original_max = config.rope_original_max_position_embeddings

    attention_factor = config.rope_attention_factor
    if attention_factor is None:
        if config.rope_mscale and config.rope_mscale_all_dim:
            attention_factor = _yarn_get_mscale(factor, config.rope_mscale) / _yarn_get_mscale(
                factor, config.rope_mscale_all_dim
            )
        else:
            attention_factor = _yarn_get_mscale(factor)

    def correction_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low = correction_dim(config.rope_beta_fast)
    high = correction_dim(config.rope_beta_slow)
    if config.rope_truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001

    pair_index = jnp.arange(dim // 2, dtype=jnp.float32)
    ramp = jnp.clip((pair_index - low) / (high - low), 0.0, 1.0)
    pos_freq = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freq
    inv_freq_interpolation = 1.0 / (factor * pos_freq)
    inv_freq = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)
    return inv_freq, attention_factor


def _deepseek_cos_sin(position_ids: jnp.ndarray, config, dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build default or YaRN rotary tables for the decoupled MLA RoPE slice."""
    rope_dim = config.qk_rope_head_dim
    if config.rope_type == "yarn":
        inv_freq, attention_factor = _deepseek_yarn_inv_freq(config)
    else:
        inv_freq = 1.0 / (config.rope_theta ** (jnp.arange(0, rope_dim, 2, dtype=jnp.float32) / rope_dim))
        attention_factor = 1.0
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return (jnp.cos(emb) * attention_factor).astype(dtype), (jnp.sin(emb) * attention_factor).astype(dtype)


class DeepseekDecoder(Block):
    """A single DeepSeek-V3 decoder layer (MLA + dense-or-MoE FFN)."""

    use_moe: bool = False
    moe_implementation: str = "auto"

    def setup(self):
        cfg = self.config
        self.self_attn = MLAAttention(cfg, dtype=self.dtype)
        self.mlp = (
            DeepseekMoE(cfg, dtype=self.dtype, moe_implementation=self.moe_implementation)
            if self.use_moe
            else LlamaMLP(cfg, dtype=self.dtype)
        )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps, dtype=self.dtype, upcast=True)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps, dtype=self.dtype, upcast=True)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos_sin: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
    ) -> DecoderOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = self.self_attn(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + output.attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual

        return DecoderOutput(
            hidden_states=hidden_states,
            kv_cache=output.kv_cache,
            attention_weight=output.attention_weight,
        )


class DeepseekModel(Block):
    """DeepSeek-V3 backbone: token embedding, MLA decoder stack, final norm."""

    moe_implementation: str = "auto"

    def setup(self):
        cfg = self.config
        if not (cfg.is_mla and cfg.is_moe):
            raise ValueError("DeepseekModel requires an MLA + MoE config.")
        if cfg.architecture == "kimi_linear" or cfg.mla_use_nope or cfg.kda_layers:
            raise ValueError("Kimi Linear configs require KimiLinearModel, not DeepseekModel.")
        self.embed_tokens = Embed(num_embeddings=cfg.vocab_size, features=self.hidden_size, dtype=self.dtype)
        self.layers = [
            DeepseekDecoder(
                cfg,
                dtype=self.dtype,
                use_moe=cfg.is_moe_layer(idx),
                moe_implementation=self.moe_implementation,
            )
            for idx in range(self.num_layers)
        ]
        self.norm = RMSNorm(self.hidden_size, eps=self.norm_eps, dtype=self.dtype, upcast=True)

    def _resolve_position_ids(self, position_ids, input_ids, kv_caches):
        if position_ids is not None:
            return position_ids
        bsz, seq_len = input_ids.shape
        offsets = jnp.arange(seq_len)[None]
        if kv_caches is not None:
            for kv_cache in kv_caches:
                if kv_cache is not None and kv_cache.pos_id is not None:
                    # Continue numbering after the cached positions (per batch row).
                    return kv_cache.next_pos_id + offsets
        return jnp.broadcast_to(offsets, (bsz, seq_len))

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[tuple[KVCache, ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> BaseModelOutputWithCache:
        cfg = self.config
        use_cache, output_attentions, output_hidden_states = normalize_model_output_flags(
            use_cache, output_attentions, output_hidden_states
        )
        input_ids, attention_mask, kv_caches = prepare_model_inputs(
            input_ids, attention_mask, kv_caches, self.num_layers, vocab_size=cfg.vocab_size
        )
        position_ids = prepare_position_ids(position_ids, input_ids, cfg.max_position_embeddings)
        resolved_position_ids = self._resolve_position_ids(position_ids, input_ids, kv_caches)

        if attention_mask is None:
            attention_mask = prepare_default_attention_mask(input_ids, kv_caches)

        input_ids = jnp.where(input_ids < 0, 0, input_ids)
        inputs_embeds = self.embed_tokens(input_ids).astype(self.dtype)
        hidden_states = inputs_embeds

        cos_sin = _deepseek_cos_sin(resolved_position_ids, cfg, self.dtype)

        all_hidden_states = []
        all_self_attns = []
        next_kv_caches = []
        for idx, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            kv_cache = None if kv_caches is None else kv_caches[idx]
            output = decoder_layer(
                hidden_states,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
            hidden_states, kv_cache, attn_weight = (output.hidden_states, output.kv_cache, output.attention_weight)
            all_self_attns.append(attn_weight)
            next_kv_caches.append(kv_cache)

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        return BaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            kv_caches=tuple(next_kv_caches) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attention_weights=tuple(all_self_attns) if output_attentions else None,
        )


class DeepseekModelWithHead(Block):
    """DeepSeek-V3 causal LM (backbone + untied LM head)."""

    lm_head_init: Any = nn.initializers.xavier_uniform
    lm_head_init_args: tuple = ()
    moe_implementation: str = "auto"

    def setup(self):
        self.model = DeepseekModel(self.config, dtype=self.dtype, moe_implementation=self.moe_implementation)
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=self.lm_head_init,
            kernel_init_args=self.lm_head_init_args,
            with_logical_partitioning=True,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[tuple[KVCache, ...]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        keep_last_n_logits: int = 0,
        logits_positions: Optional[jnp.ndarray] = None,
    ) -> CausalLMOutputWithCache:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = slice_last_n_logits_hidden_states(
            outputs.last_hidden_state,
            keep_last_n_logits,
            logits_positions=logits_positions,
        )
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithCache(
            logits=logits,
            kv_caches=outputs.kv_caches,
            hidden_states=outputs.hidden_states,
            attention_weights=outputs.attention_weights,
        )
