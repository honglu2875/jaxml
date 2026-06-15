from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jaxml.hf_utils import (
    load_gemma_from_hf,
    load_llama_from_hf,
    load_model_from_hf,
    load_neox_from_hf,
    to_gemma_jax_params,
    to_llama_jax_params,
    to_neox_jax_params,
)

pytestmark = pytest.mark.milestone


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


def test_to_neox_jax_params_rejects_rewrite_collisions():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=None)
    model = FakeModel(config)
    model.states = {
        "gpt_neox.layers.0.attention.dense.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "gpt_neox.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
    }

    with pytest.raises(ValueError, match="map to the same jaxml destination"):
        to_neox_jax_params(model, dtype=torch.float32)


@pytest.mark.parametrize(
    "name,exception,match",
    [
        ("", ValueError, "non-empty"),
        ("   ", ValueError, "non-empty"),
        (123, TypeError, "string or path-like"),
    ],
)
def test_load_model_from_hf_rejects_invalid_name_before_architecture_inference(monkeypatch, name, exception, match):
    calls = []

    def fake_infer(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("_infer_hf_architecture should not be called for invalid name.")

    monkeypatch.setattr("jaxml.hf_utils._infer_hf_architecture", fake_infer)

    with pytest.raises(exception, match=match):
        load_model_from_hf(name)

    assert calls == []


def test_load_model_from_hf_accepts_pathlike_name_before_dispatch(monkeypatch):
    calls = []

    def fake_load_llama(name, dtype, **kwargs):
        calls.append((name, dtype, kwargs))
        return "model", {"params": {}}

    monkeypatch.setattr("jaxml.hf_utils.load_llama_from_hf", fake_load_llama)

    model, params = load_model_from_hf(Path("local-model"), architecture="llama", dtype="float32", local_files_only=True)

    assert model == "model"
    assert params == {"params": {}}
    assert calls == [("local-model", "float32", {"local_files_only": True})]


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


@pytest.mark.parametrize(
    "name,exception,match",
    [
        ("", ValueError, "non-empty"),
        ("   ", ValueError, "non-empty"),
        (123, TypeError, "string or path-like"),
    ],
)
def test_direct_hf_loader_rejects_invalid_name_before_model_loading(monkeypatch, name, exception, match):
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("from_pretrained should not be called for invalid name.")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)

    with pytest.raises(exception, match=match):
        load_llama_from_hf(name)

    assert calls == []


@pytest.mark.parametrize(
    "loader,model_type,match",
    [
        (load_llama_from_hf, "gpt_neox", "Llama model_type"),
        (load_neox_from_hf, "llama", "GPT-NeoX model_type"),
    ],
)
def test_direct_hf_loader_rejects_mismatched_model_type_before_param_conversion(monkeypatch, loader, model_type, match):
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(config=SimpleNamespace(model_type=model_type), state_dict=lambda: {})

    def fake_to_params(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("parameter conversion should not run for mismatched architecture.")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr("jaxml.hf_utils.to_llama_jax_params", fake_to_params)
    monkeypatch.setattr("jaxml.hf_utils.to_neox_jax_params", fake_to_params)

    with pytest.raises(ValueError, match=match):
        loader("some/model")

    assert calls == []


def test_gemma_hf_loader_rejects_mismatched_wrapper_model_type_before_language_model_access(monkeypatch):
    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)

    with pytest.raises(ValueError, match="Gemma model_type"):
        load_gemma_from_hf("some/model")


def test_gemma_hf_loader_rejects_missing_language_model(monkeypatch):
    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)

    with pytest.raises(ValueError, match="language_model"):
        load_gemma_from_hf("some/model")


def test_gemma_hf_loader_rejects_mismatched_text_model_type_before_param_conversion(monkeypatch):
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            language_model = SimpleNamespace(config=SimpleNamespace(model_type="llama"), state_dict=lambda: {})
            return SimpleNamespace(config=SimpleNamespace(model_type="gemma3"), language_model=language_model)

    def fake_to_params(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("parameter conversion should not run for mismatched Gemma text model.")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr("jaxml.hf_utils.to_gemma_jax_params", fake_to_params)

    with pytest.raises(ValueError, match="Gemma text model_type"):
        load_gemma_from_hf("some/model")

    assert calls == []


def test_load_model_from_hf_auto_accepts_gemma_text_config(monkeypatch):
    calls = []

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(model_type="gemma3_text")

    def fake_load_gemma(name, dtype, **kwargs):
        calls.append((name, dtype, kwargs))
        return "model", {"params": {}}

    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr("jaxml.hf_utils.load_gemma_from_hf", fake_load_gemma)

    model, params = load_model_from_hf("some/model", dtype="float32", local_files_only=True)

    assert model == "model"
    assert params == {"params": {}}
    assert calls == [("some/model", "float32", {"local_files_only": True})]
