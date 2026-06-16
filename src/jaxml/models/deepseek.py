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

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.partitioning import with_sharding_constraint

from ..nn.linear import DenseGeneral
from ..nn.module import Block


class DeepseekMoE(Block):
    """Fine-grained mixture-of-experts FFN with shared experts (DeepSeek-V3).

    Routed experts are stored as stacked 3D kernels (matching the reference
    `DeepseekV3NaiveMoe` layout) and evaluated densely for every token; the
    routing weights are zero for non-selected experts, so the dense weighted sum
    is identical to the reference's gather/index-add while staying jit-friendly.
    Dense evaluation costs ``n_routed_experts`` x the active compute, which is
    fine for the small configs used in tests; production-scale sparse dispatch is
    a separate follow-up.
    """

    kernel_init: Any = nn.initializers.normal(stddev=0.02)

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

    def route(self, hidden_flat: jnp.ndarray) -> jnp.ndarray:
        """Return a dense (tokens, n_routed_experts) routing-weight matrix.

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

        routing = jnp.zeros((T, E), dtype=jnp.float32).at[rows, topk_idx].set(topk_weights.astype(jnp.float32))
        return routing

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        bsz, seq_len, H = x.shape
        inter = self.moe_intermediate_size
        hidden_flat = x.reshape(-1, H)

        routing = self.route(hidden_flat)  # (T, E), float32

        xd = hidden_flat.astype(self.dtype)
        gate_up = jnp.einsum("th,ehf->tef", xd, self.gate_up_kernel)  # (T, E, 2I)
        gate, up = gate_up[..., :inter], gate_up[..., inter:]
        expert_hidden = jax.nn.silu(gate) * up
        expert_out = jnp.einsum("tei,eih->teh", expert_hidden, self.down_kernel)  # (T, E, H)
        routed = jnp.einsum("te,teh->th", routing.astype(self.dtype), expert_out)  # (T, H)
        routed = routed.reshape(bsz, seq_len, H)

        shared = self.shared_down_proj(jax.nn.silu(self.shared_gate_proj(x)) * self.shared_up_proj(x))
        return routed + shared
