import json

import jax
import numpy as np
import pytest
import torch

import jaxml.hf_utils as hf_utils
from jaxml.config import ModelConfig
from jaxml.hf_utils import (
    _infer_hf_architecture,
    _load_qwen35_safetensors_source,
    _normalize_hf_architecture,
    _qwen35_text_parameter_count,
    _warn_for_oversized_qwen35_float32,
    load_qwen36_from_hf,
    to_qwen35_jax_params,
)

pytestmark = pytest.mark.milestone


def _tiny_qwen_text_config(**overrides):
    from transformers import Qwen3_5MoeTextConfig

    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.5,
        },
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24,
        "num_experts": 8,
        "num_experts_per_tok": 2,
    }
    values.update(overrides)
    return Qwen3_5MoeTextConfig(**values)


def test_qwen35_text_and_outer_configs_normalize_for_qwen36():
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

    text_config = _tiny_qwen_text_config()
    outer_config = Qwen3_5MoeConfig(text_config=text_config.to_dict())

    text = ModelConfig.from_hf(text_config)
    outer = ModelConfig.from_hf(outer_config)

    assert text == outer
    assert text.architecture == "qwen3_5_moe"
    assert text.intermediate_size == 24
    assert text.rotary_pct == 0.5
    assert text.layer_types == ("linear_attention", "linear_attention", "linear_attention", "full_attention")
    assert text.is_linear_attention_layer(0)
    assert not text.is_linear_attention_layer(3)
    assert text.linear_key_head_dim == 8
    assert text.linear_value_head_dim == 8
    assert text.linear_num_key_heads == 4
    assert text.linear_num_value_heads == 4
    assert text.n_routed_experts == 8
    assert text.num_experts_per_tok == 2
    assert text.moe_intermediate_size == 16
    assert text.shared_expert_intermediate_size == 24

    release = ModelConfig.from_hf(Qwen3_5MoeTextConfig())
    assert _qwen35_text_parameter_count(release) == 34_660_610_688
    with pytest.warns(RuntimeWarning, match="128 GiB aggregate HBM"):
        _warn_for_oversized_qwen35_float32(release, "float32")


def test_qwen36_aliases_and_both_internal_model_types_dispatch(tmp_path):
    for alias in (
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3-5",
        "qwen35",
        "qwen3_6",
        "qwen3-6",
        "Qwen3.6",
        "qwen36",
        "qwen",
    ):
        assert _normalize_hf_architecture(alias) == "qwen3_5"

    text_config = _tiny_qwen_text_config()
    text_config.to_json_file(tmp_path / "config.json")
    assert _infer_hf_architecture(str(tmp_path), local_files_only=True) == "qwen3_5"

    outer_config = {
        "model_type": "qwen3_5_moe",
        "text_config": text_config.to_dict(),
    }
    (tmp_path / "config.json").write_text(json.dumps(outer_config))
    assert _infer_hf_architecture(str(tmp_path), local_files_only=True) == "qwen3_5"


def test_qwen_outer_safetensors_source_is_lazy_and_text_only(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3_5MoeConfig

    outer_config = Qwen3_5MoeConfig(text_config=_tiny_qwen_text_config().to_dict())
    outer_config.to_json_file(tmp_path / "config.json")
    text_weight = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    save_file(
        {
            "model.language_model.embed_tokens.weight": text_weight,
            "model.visual.patch_embed.weight": torch.ones((2, 2)),
            "mtp.layers.0.weight": torch.ones((2, 2)),
            "lm_head.weight": text_weight + 1,
        },
        tmp_path / "model.safetensors",
    )

    source = _load_qwen35_safetensors_source(str(tmp_path))
    state_dict = source.state_dict()

    assert source.config.model_type == "qwen3_5_moe"
    assert set(state_dict) == {"model.language_model.embed_tokens.weight", "lm_head.weight"}
    assert state_dict._loaded_name is None
    assert np.array_equal(state_dict["model.language_model.embed_tokens.weight"].numpy(), text_weight.numpy())
    assert state_dict._loaded_name == "model.safetensors"

    quantized_config = outer_config.to_dict()
    quantized_config["quantization_config"] = {"quant_method": "fp8"}
    (tmp_path / "config.json").write_text(json.dumps(quantized_config))
    with pytest.raises(NotImplementedError, match="scale-aware FP8 conversion"):
        _load_qwen35_safetensors_source(str(tmp_path))


def test_qwen36_release_loader_alias_and_dispatch(monkeypatch):
    calls = []

    def fake_loader(name, dtype="float32", **kwargs):
        calls.append((name, dtype, kwargs))
        return "model", "params"

    monkeypatch.setattr(hf_utils, "load_qwen35_from_hf", fake_loader)

    assert hf_utils.load_qwen36_from_hf("Qwen/default") == ("model", "params")
    assert hf_utils.load_qwen36_from_hf("Qwen/release", dtype="bfloat16") == ("model", "params")
    assert hf_utils.load_model_from_hf(
        "Qwen/release",
        architecture="qwen3-6",
        dtype="float16",
        local_files_only=True,
    ) == ("model", "params")
    assert calls == [
        ("Qwen/default", "bfloat16", {}),
        ("Qwen/release", "bfloat16", {}),
        ("Qwen/release", "float16", {"local_files_only": True}),
    ]


def test_qwen_converter_and_outer_loader_map_to_the_same_tree(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM

    from jaxml.models.qwen3_5 import Qwen35ModelWithHead

    text_config = _tiny_qwen_text_config(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    torch.manual_seed(7)
    hf_model = Qwen3_5MoeForCausalLM(text_config)
    text_params = to_qwen35_jax_params(hf_model, dtype="float16")

    linear = text_params["params"]["model"]["layers_0"]["linear_attn"]
    full = text_params["params"]["model"]["layers_1"]["self_attn"]
    moe = text_params["params"]["model"]["layers_1"]["mlp"]
    assert linear["conv1d"]["weight"].shape == (96, 4)
    assert linear["in_proj_z"]["kernel"].shape == (32, 4, 8)
    assert linear["A_log"].dtype == np.float32
    assert linear["dt_bias"].dtype == np.float32
    assert full["q_proj"]["kernel"].shape == (32, 4, 2, 8)
    assert full["k_proj"]["kernel"].shape == (32, 2, 8)
    assert moe["gate_kernel"].dtype == np.float32
    assert moe["gate_up_kernel"].shape == (8, 32, 32)
    assert moe["shared_expert"]["down_proj"]["kernel"].shape == (24, 32)
    assert np.array_equal(
        full["q_norm"]["weight"],
        hf_model.state_dict()["model.layers.1.self_attn.q_norm.weight"].detach().numpy().astype(np.float16),
    )

    outer_config = Qwen3_5MoeConfig(text_config=text_config.to_dict())
    outer_states = {
        ("model.language_model." + key.removeprefix("model.") if key.startswith("model.") else key): value
        for key, value in hf_model.state_dict().items()
    }
    outer_config.to_json_file(tmp_path / "config.json")
    save_file(outer_states, tmp_path / "model.safetensors")
    loaded_model, outer_params = load_qwen36_from_hf(str(tmp_path), dtype="float16")

    assert isinstance(loaded_model, Qwen35ModelWithHead)
    assert jax.tree.structure(outer_params) == jax.tree.structure(text_params)
    assert all(np.array_equal(outer, text) for outer, text in zip(jax.tree.leaves(outer_params), jax.tree.leaves(text_params)))
