import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.inference_engine.sampling import (
    SamplingMethod,
    min_p_filtering,
    normalize_sampling_params,
    top_k_filtering,
    top_p_filtering,
)


def _is_kept(filtered):
    return ~np.isneginf(np.array(filtered))


def test_normalize_sampling_params_clips_bounds():
    params = normalize_sampling_params(top_k=-3, top_p=1.5, min_p=-0.5, temp=-1.0)

    assert params.top_k == 0
    assert params.top_p == 1.0
    assert params.min_p == 0.0
    assert params.temp == 0.0


@pytest.mark.parametrize("top_k", [True, 1.5])
def test_normalize_sampling_params_rejects_non_integer_top_k(top_k):
    with pytest.raises(TypeError, match="top_k must be an integer"):
        normalize_sampling_params(top_k=top_k, top_p=1.0, min_p=0.0, temp=1.0)


@pytest.mark.parametrize("name", ["top_p", "min_p", "temp"])
@pytest.mark.parametrize("value", [True, "1.0"])
def test_normalize_sampling_params_rejects_non_real_values(name, value):
    kwargs = {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "temp": 1.0}
    kwargs[name] = value

    with pytest.raises(TypeError, match=f"{name} must be a real number"):
        normalize_sampling_params(**kwargs)


@pytest.mark.parametrize("name", ["top_p", "min_p", "temp"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_normalize_sampling_params_rejects_non_finite_values(name, value):
    kwargs = {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "temp": 1.0}
    kwargs[name] = value

    with pytest.raises(ValueError, match=f"{name} must be finite"):
        normalize_sampling_params(**kwargs)


def test_top_k_filtering_keeps_top_tokens_per_batch_row():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array(
        [
            [[0.0, 1.0, 5.0, 2.0]],
            [[4.0, 1.0, 0.0, 3.0]],
        ]
    )

    filtered = top_k_filtering(rng, logits, 2, 1.0, 0.0, 1.0)

    assert np.array_equal(
        _is_kept(filtered),
        np.array(
            [
                [[False, False, True, True]],
                [[True, False, False, True]],
            ]
        ),
    )


def test_top_k_filtering_caps_k_to_vocab_size():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    filtered = top_k_filtering(rng, logits, 8, 1.0, 0.0, 1.0)

    assert np.array_equal(_is_kept(filtered), np.array([[[True, True, True, True]]]))


def test_top_k_filtering_is_noop_when_disabled():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    filtered = top_k_filtering(rng, logits, 0, 1.0, 0.0, 1.0)

    assert np.array_equal(np.array(filtered), np.array(logits))


def test_top_p_filtering_uses_sorted_cutoff_per_batch_row():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array(
        [
            [[0.0, 1.0, 5.0, 2.0]],
            [[4.0, 1.0, 0.0, 3.0]],
        ]
    )

    filtered = top_p_filtering(rng, logits, 0, 0.7, 0.0, 1.0)

    assert np.array_equal(
        _is_kept(filtered),
        np.array(
            [
                [[False, False, True, False]],
                [[True, False, False, True]],
            ]
        ),
    )


def test_top_p_filtering_is_noop_at_or_above_one():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    filtered = top_p_filtering(rng, logits, 0, 1.5, 0.0, 1.0)

    assert np.array_equal(np.array(filtered), np.array(logits))


def test_top_p_filtering_keeps_max_logits_at_zero():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 5.0, 5.0, 2.0]]])

    filtered = top_p_filtering(rng, logits, 0, 0.0, 0.0, 1.0)

    assert np.array_equal(_is_kept(filtered), np.array([[[False, True, True, False]]]))


def test_min_p_filtering_falls_back_per_batch_row_when_no_token_qualifies():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array(
        [
            [[10.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )

    filtered = min_p_filtering(rng, logits, 0, 1.0, 0.9, 1.0)

    assert np.array_equal(
        _is_kept(filtered),
        np.array(
            [
                [[True, False, False]],
                [[True, True, True]],
            ]
        ),
    )


def test_sampling_method_pipeline_handles_batched_logits():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array(
        [
            [[0.0, 1.0, 5.0, 2.0]],
            [[4.0, 1.0, 0.0, 3.0]],
        ]
    )
    sampling_fn = SamplingMethod.from_values(top_k=2, top_p=0.9, min_p=0.0, temp=1.0).get_sampling_fn()

    sampled = sampling_fn(rng, logits, 2, 0.9, 0.0, 1.0)

    assert sampled.shape == (2, 1)
    assert np.all(np.array(sampled[0]) != 0)
    assert np.all(np.array(sampled[1]) != 1)


def test_sampling_method_pipeline_handles_top_k_larger_than_vocab():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])
    sampling_fn = SamplingMethod.from_values(top_k=8, top_p=1.0, min_p=0.0, temp=1.0).get_sampling_fn()

    sampled = sampling_fn(rng, logits, 8, 1.0, 0.0, 1.0)

    assert sampled.shape == (1, 1)
    assert 0 <= int(sampled[0, 0]) < logits.shape[-1]
