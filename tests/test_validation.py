import jax
import jax.numpy as jnp
import pytest

from jaxml._validation import contains_tracer, validate_finite_values

pytestmark = pytest.mark.critical


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
def test_validate_finite_values_rejects_concrete_non_finite_values(value):
    values = jnp.array([0.0, value], dtype=jnp.float32)

    with pytest.raises(ValueError, match="activations must contain only finite values"):
        validate_finite_values("activations", values)


def test_validate_finite_values_accepts_concrete_finite_values():
    validate_finite_values("activations", jnp.array([0.0, 1.0], dtype=jnp.float32))


def test_validate_finite_values_accepts_traced_values():
    @jax.jit
    def apply(values):
        validate_finite_values("activations", values)
        return values + 1

    output = apply(jnp.array([0.0, 1.0], dtype=jnp.float32))

    assert jnp.array_equal(output, jnp.array([1.0, 2.0], dtype=jnp.float32))


def test_contains_tracer_identifies_traced_tree_leaves():
    @jax.jit
    def apply(values):
        return contains_tracer({"values": values})

    assert bool(apply(jnp.array([0.0], dtype=jnp.float32)))
