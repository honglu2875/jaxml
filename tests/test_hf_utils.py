from types import SimpleNamespace

import torch

from jaxml.hf_utils import to_gemma_jax_params, to_llama_jax_params, to_neox_jax_params


class FakeModel:
    def __init__(self, config):
        self.config = config

    def state_dict(self):
        return {
            "q_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "k_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "v_proj.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        }


def test_to_llama_jax_params_falls_back_when_head_dim_is_missing():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2)
    params = to_llama_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["q_proj"]["kernel"].shape == (8, 1, 4)


def test_to_gemma_jax_params_falls_back_when_head_dim_is_none():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=None)
    params = to_gemma_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["k_proj"]["kernel"].shape == (8, 1, 4)


def test_to_neox_jax_params_uses_head_dim_fallback():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=None)
    params = to_neox_jax_params(FakeModel(config), dtype=torch.float32)

    assert params["params"]["q_proj"]["kernel"].shape == (8, 1, 4)
