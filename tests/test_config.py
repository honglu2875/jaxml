import pytest

from jaxml.config import ModelConfig


def test_model_config_from_hf_accepts_supported_config_subclasses():
    from transformers import LlamaConfig

    class CustomLlamaConfig(LlamaConfig):
        pass

    hf_config = CustomLlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
        rope_theta=5000.0,
        rms_norm_eps=1e-5,
    )

    config = ModelConfig.from_hf(hf_config)

    assert config.hidden_size == 48
    assert config.head_dim == 8
    assert config.num_heads == 6
    assert config.num_key_value_heads == 3
    assert config.intermediate_size == 144
    assert config.rope_theta == 5000.0
    assert config.norm_eps == 1e-5


def test_model_config_from_hf_maps_neox_specific_fields():
    from transformers import GPTNeoXConfig

    hf_config = GPTNeoXConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        rotary_emb_base=12345.0,
        rotary_pct=0.5,
        use_parallel_residual=False,
        attention_bias=True,
    )

    config = ModelConfig.from_hf(hf_config)

    assert config.num_key_value_heads == 6
    assert config.rope_theta == 12345.0
    assert config.rotary_pct == 0.5
    assert config.use_parallel_residual is False
    assert config.use_bias is True


def test_model_config_from_hf_maps_gemma3_specific_fields():
    from transformers import Gemma3TextConfig

    hf_config = Gemma3TextConfig(
        hidden_size=64,
        head_dim=8,
        intermediate_size=144,
        num_hidden_layers=4,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
        rope_scaling={"factor": 8.0, "rope_type": "linear"},
        sliding_window=32,
        sliding_window_pattern=2,
    )

    config = ModelConfig.from_hf(hf_config)

    assert config.hidden_size == 64
    assert config.head_dim == 8
    assert config.num_key_value_heads == 3
    assert config.rope_scale == 8.0
    assert config.sliding_window == 32
    assert config.sliding_window_pattern == 2
    assert config.attn_scale == hf_config.query_pre_attn_scalar**-0.5


def test_model_config_from_hf_rejects_unsupported_config():
    from transformers import MistralConfig

    with pytest.raises(ValueError, match="Unsupported config class"):
        ModelConfig.from_hf(MistralConfig())
