from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jaxml.hf_utils import (
    load_llama_from_hf,
    load_model_from_hf,
    to_gemma_jax_params,
    to_llama_jax_params,
    to_neox_jax_params,
)


class FakeModel:
    def __init__(self, config):
        self.config = config
        self.states = {
            "q_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "k_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "v_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        }

    def state_dict(self):
        return self.states


def test_to_llama_jax_params_falls_back_when_head_dim_is_missing():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2)
    params = to_llama_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["q_proj"]["kernel"].shape == (8, 1, 4)


def test_to_gemma_jax_params_falls_back_when_head_dim_is_none():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=None)
    params = to_gemma_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["k_proj"]["kernel"].shape == (8, 1, 4)


def test_to_llama_jax_params_accepts_integer_like_head_dim():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=np.int64(4))
    params = to_llama_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["q_proj"]["kernel"].shape == (8, 1, 4)


@pytest.mark.parametrize(
    ("head_dim", "exception", "match"),
    [
        (0, ValueError, "head_dim must be positive"),
        (-1, ValueError, "head_dim must be positive"),
        (True, TypeError, "head_dim must be an integer"),
        (np.bool_(True), TypeError, "head_dim must be an integer"),
        (1.5, TypeError, "head_dim must be an integer"),
    ],
)
def test_to_llama_jax_params_rejects_invalid_head_dim(head_dim, exception, match):
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=head_dim)

    with pytest.raises(exception, match=match):
        to_llama_jax_params(FakeModel(config), dtype=torch.float32)


def test_to_llama_jax_params_rejects_non_divisible_head_dim_fallback():
    config = SimpleNamespace(hidden_size=10, num_attention_heads=3)

    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        to_llama_jax_params(FakeModel(config), dtype=torch.float32)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (SimpleNamespace(hidden_size=8.0, num_attention_heads=2), "hidden_size must be an integer"),
        (SimpleNamespace(hidden_size=8, num_attention_heads=True), "num_attention_heads must be an integer"),
        (SimpleNamespace(hidden_size=0, num_attention_heads=2), "hidden_size must be positive"),
        (SimpleNamespace(hidden_size=8, num_attention_heads=0), "num_attention_heads must be positive"),
    ],
)
def test_to_llama_jax_params_rejects_invalid_head_dim_fallback(config, match):
    with pytest.raises((TypeError, ValueError), match=match):
        to_llama_jax_params(FakeModel(config), dtype=torch.float32)


def test_to_neox_jax_params_uses_head_dim_fallback():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=None)
    params = to_neox_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["q_proj"]["kernel"].shape == (8, 1, 4)


def test_to_neox_jax_params_does_not_mutate_state_dict_keys():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=None)
    model = FakeModel(config)
    model.states = {
        "gpt_neox.embed_in.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        "gpt_neox.final_layer_norm.weight": torch.arange(8, dtype=torch.float32),
    }

    params = to_neox_jax_params(model, dtype=torch.float32)

    assert "embed_tokens" in params["params"]["gpt_neox"]
    assert "norm" in params["params"]["gpt_neox"]
    assert set(model.states) == {"gpt_neox.embed_in.weight", "gpt_neox.final_layer_norm.weight"}


@pytest.mark.parametrize(
    "dtype,exception,match",
    [
        (np.float32, TypeError, "Expected dtype"),
        ("bad", ValueError, "Unsupported dtype"),
        (torch.int32, ValueError, "Unsupported dtype"),
    ],
)
def test_load_model_from_hf_rejects_invalid_dtype_before_architecture_inference(monkeypatch, dtype, exception, match):
    calls = []

    def fake_infer(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("_infer_hf_architecture should not be called for invalid dtype.")

    monkeypatch.setattr("jaxml.hf_utils._infer_hf_architecture", fake_infer)

    with pytest.raises(exception, match=match):
        load_model_from_hf("some/model", dtype=dtype)

    assert calls == []


def test_direct_hf_loader_rejects_invalid_dtype_before_model_loading(monkeypatch):
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("from_pretrained should not be called for invalid dtype.")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)

    with pytest.raises(TypeError, match="Expected dtype"):
        load_llama_from_hf("some/model", dtype=np.float32)

    assert calls == []
