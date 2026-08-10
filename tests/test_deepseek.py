#  Copyright 2024 Honglu Fan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import linen as nn

from jaxml.cache import MLACache
from jaxml.config import ModelConfig
from jaxml.hf_utils import load_model_from_hf, to_deepseek_jax_params
from jaxml.models.deepseek import DeepseekModelWithHead, DeepseekMoE, MLAAttention

pytestmark = pytest.mark.milestone


def _tiny_hf_deepseek_config(**overrides):
    from transformers import DeepseekV3Config

    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=4,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=0,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=100,
        max_position_embeddings=64,
    )
    kwargs.update(overrides)
    return DeepseekV3Config(**kwargs)


def _np(tensor):
    return tensor.detach().cpu().float().numpy()


def _moe_params_from_hf(hf_moe):
    """Map an HF DeepseekV3MoE state dict to the jaxml DeepseekMoE param tree.

    HF stores routed experts as stacked 3D tensors with the torch ``Linear``
    convention (out, in); jaxml expects ``x @ kernel``, so each expert kernel is
    transposed on its last two axes.
    """
    sd = hf_moe.state_dict()
    params = {
        "gate_kernel": _np(sd["gate.weight"]).T,
        "e_score_correction_bias": _np(sd["gate.e_score_correction_bias"]),
        "gate_up_kernel": _np(sd["experts.gate_up_proj"]).transpose(0, 2, 1),
        "down_kernel": _np(sd["experts.down_proj"]).transpose(0, 2, 1),
        "shared_gate_proj": {"kernel": _np(sd["shared_experts.gate_proj.weight"]).T},
        "shared_up_proj": {"kernel": _np(sd["shared_experts.up_proj.weight"]).T},
        "shared_down_proj": {"kernel": _np(sd["shared_experts.down_proj.weight"]).T},
    }
    return {"params": jax.tree.map(jnp.asarray, params)}


def test_deepseek_moe_matches_hf():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_deepseek_config()
    hf_moe = DeepseekV3MoE(hf_cfg).eval()
    # DeepseekV3NaiveMoe allocates expert weights with torch.empty (uninitialized);
    # initialize every parameter + the correction-bias buffer to finite values.
    with torch.no_grad():
        for param in hf_moe.parameters():
            param.normal_(0.0, 0.02)
        hf_moe.gate.e_score_correction_bias.normal_(0.0, 0.02)

    config = ModelConfig.from_hf(hf_cfg)
    model = DeepseekMoE(config, dtype=jnp.float32)

    bs, seq_len = 2, 5
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
        params = _moe_params_from_hf(hf_moe)
        # Converted params must match a fresh init leaf-for-leaf in name and shape
        # (apply would also raise on a mismatch, but check explicitly for clarity).
        init_shapes = jax.tree.map(jnp.shape, nn.meta.unbox(model.init(jax.random.PRNGKey(1), x)))
        assert init_shapes == jax.tree.map(jnp.shape, params)

        y = model.apply(params, x)
        with torch.no_grad():
            y2 = hf_moe(torch.tensor(np.array(x))).numpy()

    assert np.allclose(y, y2, atol=1e-5)


def _mla_params_from_hf(hf_attn, config):
    """Map an HF DeepseekV3Attention state dict to the jaxml MLAAttention tree."""
    sd = hf_attn.state_dict()
    nh, qk = config.num_heads, config.qk_head_dim
    nope, v = config.qk_nope_head_dim, config.v_head_dim
    p = {}
    if "q_proj.weight" in sd:
        p["q_proj"] = {"kernel": _np(sd["q_proj.weight"]).T.reshape(-1, nh, qk)}
    else:
        p["q_a_proj"] = {"kernel": _np(sd["q_a_proj.weight"]).T}
        p["q_a_layernorm"] = {"weight": _np(sd["q_a_layernorm.weight"])}
        p["q_b_proj"] = {"kernel": _np(sd["q_b_proj.weight"]).T.reshape(-1, nh, qk)}
    p["kv_a_proj_with_mqa"] = {"kernel": _np(sd["kv_a_proj_with_mqa.weight"]).T}
    p["kv_a_layernorm"] = {"weight": _np(sd["kv_a_layernorm.weight"])}
    p["kv_b_proj"] = {"kernel": _np(sd["kv_b_proj.weight"]).T.reshape(-1, nh, nope + v)}
    p["o_proj"] = {"kernel": _np(sd["o_proj.weight"]).T}
    return {"params": jax.tree.map(jnp.asarray, p)}


def test_mla_attention_prefill_matches_hf():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3Attention,
        DeepseekV3RotaryEmbedding,
    )

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_deepseek_config()
    hf_cfg._attn_implementation = "eager"
    hf_attn = DeepseekV3Attention(hf_cfg, layer_idx=0).eval()
    rotary = DeepseekV3RotaryEmbedding(hf_cfg)

    config = ModelConfig.from_hf(hf_cfg)
    model = MLAAttention(config, dtype=jnp.float32)

    bs, seq_len = 2, 6
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
        xt = torch.tensor(np.array(x))
        position_ids = torch.arange(seq_len)[None].expand(bs, -1)
        cos, sin = rotary(xt, position_ids)
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)[None, None]
        with torch.no_grad():
            hf_out = hf_attn(xt, position_embeddings=(cos, sin), attention_mask=causal, past_key_values=None)[0].numpy()

        params = _mla_params_from_hf(hf_attn, config)
        cos_sin = (jnp.asarray(cos.numpy()), jnp.asarray(sin.numpy()))
        out = model.apply(params, x, cos_sin).attention_output

    assert np.allclose(out, hf_out, atol=1e-5)


def test_mla_attention_decode_cache_consistency():
    """Incremental decode through compressed MLACache must match full attention."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

    hf_cfg = _tiny_hf_deepseek_config()
    rotary = DeepseekV3RotaryEmbedding(hf_cfg)
    config = ModelConfig.from_hf(hf_cfg)
    model = MLAAttention(config, dtype=jnp.float32)

    bs, seq_len = 2, 4
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
        position_ids = torch.arange(seq_len)[None].expand(bs, -1)
        cos, sin = rotary(torch.tensor(np.array(x)), position_ids)
        cos, sin = jnp.asarray(cos.numpy()), jnp.asarray(sin.numpy())

        params = model.init(jax.random.PRNGKey(1), x, (cos, sin))
        full = model.apply(params, x, (cos, sin)).attention_output

        cache = MLACache.init(max_seq_len=seq_len, dtype=jnp.float32)
        prefill = model.apply(params, x[:, : seq_len - 1], (cos[:, : seq_len - 1], sin[:, : seq_len - 1]), kv_cache=cache)
        decode = model.apply(
            params,
            x[:, seq_len - 1 :],
            (cos[:, seq_len - 1 :], sin[:, seq_len - 1 :]),
            kv_cache=prefill.kv_cache,
        )

    assert isinstance(prefill.kv_cache, MLACache)
    assert prefill.kv_cache.kv_latent.shape == (bs, seq_len, config.kv_lora_rank)
    assert prefill.kv_cache.k_rope.shape == (bs, seq_len, config.qk_rope_head_dim)
    compressed_elements = prefill.kv_cache.kv_latent.size + prefill.kv_cache.k_rope.size
    expanded_elements = 2 * bs * seq_len * config.num_heads * config.qk_head_dim
    assert compressed_elements < expanded_elements
    assert np.allclose(decode.attention_output[:, -1], full[:, -1], atol=1e-5)


def test_deepseek_model_initializes_compressed_mla_caches():
    config = ModelConfig.from_hf(_tiny_hf_deepseek_config(num_hidden_layers=3))
    model = DeepseekModelWithHead(config, dtype=jnp.bfloat16)

    caches = model.init_cache(max_seq_len=32)

    assert len(caches) == config.num_layers
    assert all(isinstance(cache, MLACache) for cache in caches)
    assert all(cache.kv_latent is None and cache.k_rope is None for cache in caches)
    assert all(cache.max_seq_len == 32 and cache.dtype == jnp.bfloat16 for cache in caches)


def _tiny_tpu_deepseek_config(**overrides):
    # megablox.gmm tiles the K/N dims by 128, so hidden_size and moe_intermediate_size
    # must be multiples of 128 for the grouped matmul.
    from transformers import DeepseekV3Config

    kwargs = dict(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=4,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=0,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        vocab_size=128,
        max_position_embeddings=64,
    )
    kwargs.update(overrides)
    return DeepseekV3Config(**kwargs)


def test_deepseek_moe_auto_uses_gmm_for_single_token_tpu_decode(monkeypatch):
    config = ModelConfig.from_hf(_tiny_tpu_deepseek_config())
    moe = DeepseekMoE(config, dtype=jnp.float32, moe_implementation="auto")
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    assert moe._resolve_implementation(seq_len=1) == "gmm"
    assert moe._resolve_implementation(seq_len=8) == "gmm"


def test_deepseek_head_logits_positions_match_full_logits_for_right_padded_batch():
    config = ModelConfig.from_hf(_tiny_hf_deepseek_config(num_hidden_layers=1, first_k_dense_replace=1))
    model = DeepseekModelWithHead(config, dtype=jnp.float32)
    input_ids = jnp.array([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    logits_positions = jnp.array([3, 1], dtype=jnp.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        params = model.init(jax.random.PRNGKey(0), input_ids, attention_mask=attention_mask)
        full = model.apply(params, input_ids, attention_mask=attention_mask).logits
        selected = model.apply(
            params,
            input_ids,
            attention_mask=attention_mask,
            logits_positions=logits_positions,
        ).logits

    assert selected.shape == (2, 1, config.vocab_size)
    assert np.allclose(selected[:, 0], full[jnp.arange(2), logits_positions])


@pytest.mark.parametrize("seq_len", [1, 8])
@pytest.mark.tpu
def test_deepseek_moe_gmm_matches_dense_on_tpu(seq_len):
    """The Pallas gmm sparse path equals the dense reference (up to TPU bf16)."""
    config = ModelConfig.from_hf(_tiny_tpu_deepseek_config())
    dense = DeepseekMoE(config, dtype=jnp.float32, moe_implementation="dense")
    gmm = DeepseekMoE(config, dtype=jnp.float32, moe_implementation="gmm")

    bs = 2
    x = jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, config.hidden_size), dtype=jnp.float32)
    params = dense.init(jax.random.PRNGKey(1), x)  # both paths share the same parameters

    y_dense = np.asarray(dense.apply(params, x))
    y_gmm = np.asarray(gmm.apply(params, x))

    assert y_gmm.shape == y_dense.shape
    assert np.allclose(y_gmm, y_dense, atol=2e-2, rtol=2e-2)


def test_deepseek_engine_generate_end_to_end(tmp_path, monkeypatch):
    """DeepSeek threads through the full engine prefill+decode generation pipeline.

    Runs on CPU (single device, dense MoE path). The engine calls the same
    ``model.apply`` path on TPU, where the dense/gmm selection is automatic
    (gmm for both multi-token prefill and single-token decode); the gmm kernel
    itself is covered by ``test_deepseek_moe_gmm_matches_dense_on_tpu``.
    """
    from jaxml.inference_engine.engine import Engine, InferenceConfig

    monkeypatch.setenv("JAXML_CACHE_DIR", str(tmp_path))  # isolate the AOT cache
    config = ModelConfig.from_hf(_tiny_hf_deepseek_config(num_hidden_layers=3, first_k_dense_replace=1))
    model = DeepseekModelWithHead(config, moe_implementation="auto")

    rng = np.random.default_rng(0)
    prompt = jnp.asarray(rng.integers(0, config.vocab_size, size=(1, 6)).astype(np.int32))
    params = model.init(jax.random.PRNGKey(0), prompt)

    engine = Engine(model, InferenceConfig(tp_size=1, dp_size=1), params, dtype=jnp.float32)
    out = engine.generate(prompt, max_new_tokens=4, temperature=0.0, include_prompt=True)

    assert out.shape == (1, 10)
    assert bool(jnp.all((out >= 0) & (out < config.vocab_size)))


def test_deepseek_for_causal_lm_matches_hf():
    """End-to-end logits parity for the whole model (embed -> dense + MoE layers -> MLA -> head)."""
    from transformers import DeepseekV3ForCausalLM

    torch.manual_seed(0)
    # num_hidden_layers=3 with first_k_dense_replace=1 exercises one dense layer and two MoE layers.
    hf_cfg = _tiny_hf_deepseek_config(num_hidden_layers=3, first_k_dense_replace=1)
    hf_cfg._attn_implementation = "eager"
    hf_model = DeepseekV3ForCausalLM(hf_cfg).eval()

    config = ModelConfig.from_hf(hf_cfg)
    model = DeepseekModelWithHead(config)

    bs, seq_len = 2, 7
    rng = np.random.default_rng(0)
    ids = rng.integers(0, config.vocab_size, size=(bs, seq_len)).astype(np.int32)
    with jax.default_device(jax.devices("cpu")[0]):
        params = to_deepseek_jax_params(hf_model, dtype="float32")
        logits = model.apply(params, jnp.asarray(ids), use_cache=False).logits
        with torch.no_grad():
            hf_logits = hf_model(input_ids=torch.tensor(ids, dtype=torch.long)).logits.numpy()

    assert logits.shape == hf_logits.shape
    assert np.allclose(logits, hf_logits, atol=1e-4)


def test_load_deepseek_from_hf_roundtrip(tmp_path):
    """The auto loader infers DeepSeek, converts weights, and reproduces HF logits."""
    from transformers import AutoModelForCausalLM, DeepseekV3ForCausalLM

    torch.manual_seed(0)
    hf_cfg = _tiny_hf_deepseek_config(num_hidden_layers=3, first_k_dense_replace=1)
    DeepseekV3ForCausalLM(hf_cfg).eval().save_pretrained(tmp_path)

    ref = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    ref.config._attn_implementation = "eager"
    model, params = load_model_from_hf(str(tmp_path), architecture="auto", dtype="float32")
    assert isinstance(model, DeepseekModelWithHead)

    rng = np.random.default_rng(1)
    ids = rng.integers(0, hf_cfg.vocab_size, size=(2, 6)).astype(np.int32)
    with jax.default_device(jax.devices("cpu")[0]):
        logits = model.apply(params, jnp.asarray(ids), use_cache=False).logits
        with torch.no_grad():
            ref_logits = ref(input_ids=torch.tensor(ids, dtype=torch.long)).logits.numpy()

    assert np.allclose(logits, ref_logits, atol=1e-4)
