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


@pytest.mark.parametrize(
    "overrides,exception,match",
    [
        ({"num_classes": 0}, ValueError, "num_classes must be positive"),
        ({"num_layers": 0}, ValueError, "num_layers must be positive"),
        ({"hidden_dim": 0}, ValueError, "hidden_dim must be positive"),
        ({"state_dim": 0}, ValueError, "state_dim must be positive"),
        ({"num_output_classes": 0}, ValueError, "num_output_classes must be positive"),
        ({"num_classes": True}, TypeError, "num_classes must be an integer"),
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
