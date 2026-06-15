import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.experimental.rnn_discrete import RNNDiscrete, RNNDiscreteConfig


def test_rnn_discrete_forward_returns_logits():
    config = RNNDiscreteConfig(num_classes=8, num_layers=2, hidden_dim=6, state_dim=4)
    model = RNNDiscrete(config=config, dtype=jnp.float32)
    input_ids = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)
    logits = model.apply(params, input_ids)

    assert logits.shape == (2, 3, config.num_classes)
    assert logits.dtype == jnp.float32
    assert np.all(np.isfinite(np.array(logits)))


def test_rnn_discrete_forward_uses_custom_output_classes():
    config = RNNDiscreteConfig(
        num_classes=8,
        num_output_classes=3,
        num_layers=1,
        hidden_dim=6,
        state_dim=4,
    )
    model = RNNDiscrete(config=config, dtype=jnp.float32)
    input_ids = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)
    logits = model.apply(params, input_ids)

    assert logits.shape == (1, 3, 3)


def test_rnn_discrete_config_accepts_numpy_scalar_values():
    config = RNNDiscreteConfig(
        num_classes=np.int64(8),
        num_layers=np.int64(2),
        hidden_dim=np.int64(6),
        state_dim=np.int64(4),
        num_output_classes=np.int64(3),
        use_bias=np.bool_(True),
    )

    assert config.num_classes == 8
    assert config.num_layers == 2
    assert config.hidden_dim == 6
    assert config.state_dim == 4
    assert config.output_classes == 3
    assert config.use_bias is True


@pytest.mark.parametrize("dtype", [None, "not-a-dtype"])
def test_rnn_discrete_rejects_invalid_dtype(dtype):
    config = RNNDiscreteConfig(num_classes=8, num_layers=1, hidden_dim=6, state_dim=4)
    model = RNNDiscrete(config=config, dtype=dtype)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    with pytest.raises(TypeError, match="dtype must be a valid JAX dtype"):
        model.init(jax.random.PRNGKey(0), input_ids)


def test_rnn_discrete_rejects_non_callable_activation():
    config = RNNDiscreteConfig(num_classes=8, num_layers=1, hidden_dim=6, state_dim=4)
    model = RNNDiscrete(config=config, dtype=jnp.float32, act_fn=None)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    with pytest.raises(TypeError, match="act_fn must be callable"):
        model.init(jax.random.PRNGKey(0), input_ids)


def test_rnn_discrete_accepts_numpy_dtype():
    config = RNNDiscreteConfig(num_classes=8, num_layers=1, hidden_dim=6, state_dim=4)
    model = RNNDiscrete(config=config, dtype=np.float32)
    input_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    params = model.init(jax.random.PRNGKey(0), input_ids)
    logits = model.apply(params, input_ids)

    assert logits.dtype == jnp.float32


@pytest.mark.parametrize(
    "overrides,exception,match",
    [
        ({"num_classes": 0}, ValueError, "num_classes must be positive"),
        ({"num_layers": 0}, ValueError, "num_layers must be positive"),
        ({"hidden_dim": 0}, ValueError, "hidden_dim must be positive"),
        ({"state_dim": 0}, ValueError, "state_dim must be positive"),
        ({"num_output_classes": 0}, ValueError, "num_output_classes must be positive"),
        ({"num_classes": True}, TypeError, "num_classes must be an integer"),
        ({"num_layers": np.bool_(True)}, TypeError, "num_layers must be an integer"),
        ({"hidden_dim": 1.5}, TypeError, "hidden_dim must be an integer"),
        ({"state_dim": "4"}, TypeError, "state_dim must be an integer"),
        ({"num_output_classes": True}, TypeError, "num_output_classes must be an integer"),
        ({"num_output_classes": np.bool_(True)}, TypeError, "num_output_classes must be an integer"),
        ({"num_output_classes": 1.5}, TypeError, "num_output_classes must be an integer"),
        ({"use_bias": 1}, TypeError, "use_bias must be a boolean"),
    ],
)
def test_rnn_discrete_config_rejects_invalid_values(overrides, exception, match):
    kwargs = {
        "num_classes": 8,
        "num_layers": 2,
        "hidden_dim": 6,
        "state_dim": 4,
    } | overrides

    with pytest.raises(exception, match=match):
        RNNDiscreteConfig(**kwargs)
