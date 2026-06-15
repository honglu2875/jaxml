import pytest
import numpy as np

from jaxml.config import ModelConfig


def _valid_config_kwargs(**overrides):
    return {
        "hidden_size": 48,
        "head_dim": 8,
        "num_heads": 6,
        "num_layers": 2,
        "max_position_embeddings": 256,
        "vocab_size": 1024,
        "use_rope": False,
    } | overrides


def test_model_config_defaults_attention_scale_from_head_dim():
    config = ModelConfig(**_valid_config_kwargs())

    assert config.attn_scale == 8**-0.5


def test_model_config_normalizes_numpy_integer_fields():
    config = ModelConfig(
        **_valid_config_kwargs(
            hidden_size=np.int64(48),
            intermediate_ratio=(np.int64(3), np.int64(1)),
            num_kv_heads=np.int64(3),
            sliding_window=np.int64(32),
            sliding_window_pattern=np.int64(2),
        )
    )

    assert config.hidden_size == 48
    assert config.intermediate_ratio == (3, 1)
    assert config.num_kv_heads == 3
    assert config.sliding_window == 32
    assert config.sliding_window_pattern == 2


def test_model_config_normalizes_numpy_boolean_fields():
    config = ModelConfig(
        **_valid_config_kwargs(
            use_bias=np.bool_(True),
            use_alibi=np.bool_(False),
            use_rope=np.bool_(True),
            upcast_alibi=np.bool_(False),
            use_parallel_residual=np.bool_(False),
        )
    )

    assert config.use_bias is True
    assert config.use_alibi is False
    assert config.use_rope is True
    assert config.upcast_alibi is False
    assert config.use_parallel_residual is False


def test_model_config_normalizes_numpy_float_fields():
    config = ModelConfig(
        **_valid_config_kwargs(
            norm_eps=np.float32(1e-5),
            attn_scale=np.float64(0.25),
            rope_theta=np.float64(5000.0),
            rope_scale=np.float32(2.0),
            rotary_pct=np.float64(0.5),
        )
    )

    assert config.norm_eps == pytest.approx(1e-5)
    assert config.attn_scale == 0.25
    assert config.rope_theta == 5000.0
    assert config.rope_scale == 2.0
    assert config.rotary_pct == 0.5


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"hidden_size": 48.0}, "hidden_size must be an integer"),
        ({"head_dim": True}, "head_dim must be an integer"),
        ({"head_dim": np.bool_(True)}, "head_dim must be an integer"),
        ({"intermediate_ratio": (8.0, 3)}, "intermediate_ratio numerator must be an integer"),
        ({"intermediate_ratio": (8, False)}, "intermediate_ratio denominator must be an integer"),
        ({"intermediate_ratio": (8, np.bool_(False))}, "intermediate_ratio denominator must be an integer"),
        ({"num_kv_heads": 3.0}, "num_kv_heads must be an integer"),
        ({"num_kv_heads": np.bool_(True)}, "num_kv_heads must be an integer"),
        ({"sliding_window": 32.0}, "sliding_window must be an integer"),
        ({"sliding_window": np.bool_(True)}, "sliding_window must be an integer"),
        ({"sliding_window_pattern": 2.0}, "sliding_window_pattern must be an integer"),
        ({"sliding_window_pattern": np.bool_(True)}, "sliding_window_pattern must be an integer"),
    ],
)
def test_model_config_rejects_non_integer_counts(overrides, match):
    with pytest.raises(TypeError, match=match):
        ModelConfig(**_valid_config_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides,exception,match",
    [
        ({"norm_eps": True}, TypeError, "norm_eps must be a real number"),
        ({"attn_scale": "0.5"}, TypeError, "attn_scale must be a real number"),
        ({"rope_theta": np.bool_(True)}, TypeError, "rope_theta must be a real number"),
        ({"rope_scale": float("nan")}, ValueError, "rope_scale must be finite"),
        ({"rotary_pct": float("inf")}, ValueError, "rotary_pct must be finite"),
        ({"attn_scale": -float("inf")}, ValueError, "attn_scale must be finite"),
    ],
)
def test_model_config_rejects_invalid_float_fields(overrides, exception, match):
    with pytest.raises(exception, match=match):
        ModelConfig(**_valid_config_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"use_bias": 1}, "use_bias must be a boolean"),
        ({"use_alibi": 1}, "use_alibi must be a boolean"),
        ({"use_rope": 1}, "use_rope must be a boolean"),
        ({"upcast_alibi": 1}, "upcast_alibi must be a boolean"),
        ({"use_parallel_residual": "true"}, "use_parallel_residual must be a boolean"),
    ],
)
def test_model_config_rejects_non_boolean_flags(overrides, match):
    with pytest.raises(TypeError, match=match):
        ModelConfig(**_valid_config_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"head_dim": 0}, "head_dim"),
        ({"num_heads": 0}, "num_heads"),
        ({"num_layers": 0}, "num_layers"),
        ({"max_position_embeddings": 0}, "max_position_embeddings"),
        ({"vocab_size": 0}, "vocab_size"),
        ({"intermediate_ratio": (8, 0)}, "intermediate_ratio"),
        ({"intermediate_ratio": (8, 3, 1)}, "intermediate_ratio"),
        ({"norm_eps": 0.0}, "norm_eps"),
        ({"num_kv_heads": 0}, "num_key_value_heads"),
        ({"num_kv_heads": 4}, "divisible"),
        ({"sliding_window": 0}, "sliding_window"),
        ({"sliding_window_pattern": 0}, "sliding_window_pattern"),
        ({"attn_scale": 0.0}, "attn_scale"),
        ({"use_alibi": True, "use_rope": True}, "AliBi and RoPE"),
        ({"rope_theta": 0.0}, "rope_theta"),
        ({"rope_scale": 0.5}, "rope_scale"),
        ({"rotary_pct": 0.0}, "rotary_pct"),
        ({"rotary_pct": 1.5}, "rotary_pct"),
    ],
)
def test_model_config_rejects_invalid_invariants(overrides, match):
    with pytest.raises(ValueError, match=match):
        ModelConfig(**_valid_config_kwargs(**overrides))


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


def test_model_config_from_hf_accepts_integer_like_head_dim():
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    hf_config.head_dim = np.int64(8)

    config = ModelConfig.from_hf(hf_config)

    assert config.head_dim == 8


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
def test_model_config_from_hf_rejects_invalid_head_dim(head_dim, exception, match):
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    hf_config.head_dim = head_dim

    with pytest.raises(exception, match=match):
        ModelConfig.from_hf(hf_config)


def test_model_config_from_hf_rejects_non_divisible_head_dim_fallback():
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=50,
        intermediate_size=150,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    if hasattr(hf_config, "head_dim"):
        del hf_config.head_dim

    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        ModelConfig.from_hf(hf_config)


@pytest.mark.parametrize(
    "field,value,exception,match",
    [
        ("hidden_size", True, TypeError, "hidden_size must be an integer"),
        ("hidden_size", 0, ValueError, "hidden_size must be positive"),
        ("intermediate_size", np.bool_(True), TypeError, "intermediate_size must be an integer"),
        ("intermediate_size", 0, ValueError, "intermediate_size must be positive"),
    ],
)
def test_model_config_from_hf_rejects_invalid_shared_sizes_before_ratio_derivation(field, value, exception, match):
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    hf_config.head_dim = 8
    setattr(hf_config, field, value)

    with pytest.raises(exception, match=match):
        ModelConfig.from_hf(hf_config)


def test_model_config_from_hf_accepts_integer_like_rope_scaling_factor():
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    hf_config.rope_scaling = {"factor": np.float64(2.0), "rope_type": "linear"}

    config = ModelConfig.from_hf(hf_config)

    assert config.rope_scale == 2.0


@pytest.mark.parametrize(
    ("rope_scaling", "exception", "match"),
    [
        ({"rope_type": "linear"}, ValueError, "rope_scaling must include a factor"),
        ("linear", TypeError, "rope_scaling must be a mapping"),
        ({"factor": True}, TypeError, "rope_scaling factor must be a real number"),
        ({"factor": float("nan")}, ValueError, "rope_scaling factor must be finite"),
        ({"factor": 0.5}, ValueError, "rope_scale must be greater than or equal to 1.0"),
    ],
)
def test_model_config_from_hf_rejects_invalid_rope_scaling_factor(rope_scaling, exception, match):
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=48,
        intermediate_size=144,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=1024,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    hf_config.rope_scaling = rope_scaling

    with pytest.raises(exception, match=match):
        ModelConfig.from_hf(hf_config)


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


def test_model_config_from_hf_rejects_boolean_gemma_sliding_window_pattern():
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
        sliding_window=32,
        sliding_window_pattern=2,
    )
    hf_config.sliding_window_pattern = True

    with pytest.raises(TypeError, match="sliding_window_pattern must be an integer"):
        ModelConfig.from_hf(hf_config)


def test_model_config_from_hf_rejects_unsupported_config():
    from transformers import MistralConfig

    with pytest.raises(ValueError, match="Unsupported config class"):
        ModelConfig.from_hf(MistralConfig())
