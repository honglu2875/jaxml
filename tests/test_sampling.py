import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.inference_engine.sampling import (
    SamplingMethod,
    greedy_fn,
    min_p_filtering,
    normalize_sampling_params,
    top_k_filtering,
    top_p_filtering,
)

pytestmark = pytest.mark.critical


def _is_kept(filtered):
    return ~np.isneginf(np.array(filtered))


def test_normalize_sampling_params_clips_bounds():
    params = normalize_sampling_params(top_k=-3, top_p=1.5, min_p=-0.5, temp=-1.0)

    assert params.top_k == 0
    assert params.top_p == 1.0
    assert params.min_p == 0.0
    assert params.temp == 0.0


@pytest.mark.parametrize("top_k", [True, np.bool_(True), 1.5])
def test_normalize_sampling_params_rejects_non_integer_top_k(top_k):
    with pytest.raises(TypeError, match="top_k must be an integer"):
        normalize_sampling_params(top_k=top_k, top_p=1.0, min_p=0.0, temp=1.0)


@pytest.mark.parametrize("name", ["top_p", "min_p", "temp"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1.0"])
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


def test_top_k_filtering_accepts_numpy_integer_k():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    filtered = top_k_filtering(rng, logits, np.int64(2), 1.0, 0.0, 1.0)

    assert np.array_equal(_is_kept(filtered), np.array([[[False, False, True, True]]]))


@pytest.mark.parametrize("top_k", [True, np.bool_(True), 1.5])
def test_top_k_filtering_rejects_non_integer_k(top_k):
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    with pytest.raises(TypeError, match="top_k must be an integer"):
        top_k_filtering(rng, logits, top_k, 1.0, 0.0, 1.0)


def test_top_k_filtering_is_noop_when_disabled():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    filtered = top_k_filtering(rng, logits, 0, 1.0, 0.0, 1.0)

    assert np.array_equal(np.array(filtered), np.array(logits))


@pytest.mark.parametrize(
    "logits,exception,match",
    [
        (jnp.array(1.0, dtype=jnp.float32), ValueError, "vocabulary axis"),
        (jnp.ones((1, 1, 0), dtype=jnp.float32), ValueError, "non-empty vocabulary"),
        (jnp.ones((1, 1, 4), dtype=jnp.int32), TypeError, "floating point"),
    ],
)
@pytest.mark.parametrize("filter_fn", [top_k_filtering, top_p_filtering, min_p_filtering, greedy_fn])
def test_sampling_functions_reject_invalid_logits(filter_fn, logits, exception, match):
    rng = jax.random.PRNGKey(0)

    with pytest.raises(exception, match=match):
        filter_fn(rng, logits, 2, 0.9, 0.1, 1.0)


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


@pytest.mark.parametrize(
    "top_p,exception,match",
    [
        (True, TypeError, "top_p must be a real number"),
        (np.bool_(True), TypeError, "top_p must be a real number"),
        ("0.9", TypeError, "top_p must be a real number"),
        (float("nan"), ValueError, "top_p must be finite"),
    ],
)
def test_top_p_filtering_rejects_invalid_top_p(top_p, exception, match):
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    with pytest.raises(exception, match=match):
        top_p_filtering(rng, logits, 0, top_p, 0.0, 1.0)


def test_top_p_filtering_keeps_max_logits_at_zero():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 5.0, 5.0, 2.0]]])

    filtered = top_p_filtering(rng, logits, 0, 0.0, 0.0, 1.0)

    assert np.array_equal(_is_kept(filtered), np.array([[[False, True, True, False]]]))


@pytest.mark.parametrize(
    "min_p,exception,match",
    [
        (True, TypeError, "min_p must be a real number"),
        (np.bool_(True), TypeError, "min_p must be a real number"),
        ("0.1", TypeError, "min_p must be a real number"),
        (float("nan"), ValueError, "min_p must be finite"),
    ],
)
def test_min_p_filtering_rejects_invalid_min_p(min_p, exception, match):
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])

    with pytest.raises(exception, match=match):
        min_p_filtering(rng, logits, 0, 1.0, min_p, 1.0)


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


def test_sampling_method_pipeline_rejects_invalid_logits():
    rng = jax.random.PRNGKey(0)
    sampling_fn = SamplingMethod.from_values(top_k=2, top_p=0.9, min_p=0.0, temp=1.0).get_sampling_fn()

    with pytest.raises(TypeError, match="floating point"):
        sampling_fn(rng, jnp.ones((1, 1, 4), dtype=jnp.int32), 2, 0.9, 0.0, 1.0)


def test_sampling_method_pipeline_rejects_non_positive_temperature_for_non_greedy_method():
    rng = jax.random.PRNGKey(0)
    logits = jnp.array([[[0.0, 1.0, 5.0, 2.0]]])
    sampling_fn = SamplingMethod(use_top_k=True, use_top_p=False, use_min_p=False, use_greedy=False).get_sampling_fn()

    with pytest.raises(ValueError, match="temp must be positive"):
        sampling_fn(rng, logits, 2, 1.0, 0.0, 0.0)


def test_sampling_method_accepts_numpy_boolean_flags():
    method = SamplingMethod(
        use_top_k=np.bool_(True),
        use_top_p=np.bool_(False),
        use_min_p=np.bool_(False),
        use_greedy=np.bool_(False),
    )

    assert method.use_top_k is True
    assert method.use_top_p is False
    assert method.use_min_p is False
    assert method.use_greedy is False


@pytest.mark.parametrize("field_name", ["use_top_k", "use_top_p", "use_min_p", "use_greedy"])
@pytest.mark.parametrize("value", [1, 0, "true"])
def test_sampling_method_rejects_non_boolean_flags(field_name, value):
    kwargs = {
        "use_top_k": False,
        "use_top_p": False,
        "use_min_p": False,
        "use_greedy": True,
    }
    kwargs[field_name] = value

    with pytest.raises(TypeError, match=f"{field_name} must be a boolean"):
        SamplingMethod(**kwargs)
