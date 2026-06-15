import jax.numpy as jnp
import numpy as np
import pytest

from jaxml._generate import generate


class RngSamplingMethod:
    def get_sampling_fn(self):
        def sample_fn(rng, logits, top_k, top_p, min_p, temp):
            token = rng[1] % logits.shape[-1]
            return jnp.full(logits.shape[:-1], token, dtype=jnp.int32)

        return sample_fn


def test_prefill_rng_is_dynamic_when_compiled_function_is_reused(monkeypatch):
    cached_fns = {}

    def fake_load_if_exists(name, hash, log=True):
        def decorator(fn):
            cached_fn = cached_fns.setdefault((name, hash), fn)

            def wrapped(*args, **kwargs):
                return cached_fn(*args, **kwargs)

            return wrapped

        return decorator

    def eval_fn(params, tokens, attention_mask=None, kv_caches=None, use_cache=True):
        del params, attention_mask, use_cache
        logits = jnp.zeros(tokens.shape + (10,), dtype=jnp.float32)
        return logits, kv_caches

    monkeypatch.setattr("jaxml._generate.load_if_exists", fake_load_if_exists)
    prompt_tokens = jnp.array([[0]], dtype=jnp.int32)

    first = generate(
        {},
        eval_fn,
        prompt_tokens,
        attention_mask=None,
        kv_caches=(),
        call_hash="same-cache-key",
        sampling_method=RngSamplingMethod(),
        seed=0,
        max_new_tokens=1,
    )
    second = generate(
        {},
        eval_fn,
        prompt_tokens,
        attention_mask=None,
        kv_caches=(),
        call_hash="same-cache-key",
        sampling_method=RngSamplingMethod(),
        seed=1,
        max_new_tokens=1,
    )

    assert first.tokens.shape == (1, 1)
    assert second.tokens.shape == (1, 1)
    assert first.tokens[0, 0] != second.tokens[0, 0]


@pytest.mark.parametrize(
    "prompt_tokens,max_new_tokens,match",
    [
        (jnp.ones((1, 2, 3), dtype=jnp.int32), 1, "1D or 2D"),
        (jnp.ones((1, 0), dtype=jnp.int32), 1, "at least one token"),
        (jnp.ones((1, 1), dtype=jnp.int32), -1, "max_new_tokens"),
    ],
)
def test_generate_rejects_invalid_inputs_before_prefill(prompt_tokens, max_new_tokens, match):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid generate inputs.")

    with pytest.raises(ValueError, match=match):
        generate(
            {},
            eval_fn,
            prompt_tokens,
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-input",
            sampling_method=RngSamplingMethod(),
            max_new_tokens=max_new_tokens,
        )


def test_generate_rejects_non_integer_prompt_tokens_before_prefill():
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid generate inputs.")

    with pytest.raises(TypeError, match="integer token ids"):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.float32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-input",
            sampling_method=RngSamplingMethod(),
            max_new_tokens=1,
        )


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"seed": True}, TypeError, "seed must be an integer"),
        ({"seed": np.bool_(True)}, TypeError, "seed must be an integer"),
        ({"seed": 1.5}, TypeError, "seed must be an integer"),
        ({"max_new_tokens": True}, TypeError, "max_new_tokens must be an integer"),
        ({"max_new_tokens": np.bool_(True)}, TypeError, "max_new_tokens must be an integer"),
        ({"max_new_tokens": 1.5}, TypeError, "max_new_tokens must be an integer"),
        ({"include_prompt": 1}, TypeError, "include_prompt must be a boolean"),
        ({"fuse_decoding": 1}, TypeError, "fuse_decoding must be a boolean"),
        ({"skip_prefill": 1}, TypeError, "skip_prefill must be a boolean"),
    ],
)
def test_generate_rejects_invalid_control_arguments_before_prefill(kwargs, exception, match):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid generate inputs.")

    generate_kwargs = {"max_new_tokens": 1, **kwargs}
    with pytest.raises(exception, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-input",
            sampling_method=RngSamplingMethod(),
            **generate_kwargs,
        )


def test_generate_accepts_numpy_scalar_control_arguments():
    def eval_fn(params, tokens, attention_mask=None, kv_caches=None, use_cache=True):
        del params, attention_mask, use_cache
        logits = jnp.zeros(tokens.shape + (10,), dtype=jnp.float32)
        return logits, kv_caches

    output = generate(
        {},
        eval_fn,
        jnp.ones((1, 1), dtype=jnp.int32),
        attention_mask=None,
        kv_caches=(),
        call_hash="numpy-scalar-controls",
        sampling_method=RngSamplingMethod(),
        seed=np.int64(0),
        max_new_tokens=np.int64(1),
        include_prompt=np.bool_(False),
        fuse_decoding=np.bool_(False),
        skip_prefill=np.bool_(False),
    )

    assert output.tokens.shape == (1, 1)


@pytest.mark.parametrize(
    "attention_mask,exception,match",
    [
        (jnp.ones((1, 1, 1), dtype=jnp.int32), ValueError, "attention_mask must be a 1D or 2D array"),
        (jnp.ones((1, 2), dtype=jnp.int32), ValueError, "attention_mask shape must match"),
        (jnp.ones((1, 1), dtype=jnp.float32), TypeError, "attention_mask must be boolean or integer"),
        (jnp.zeros((1, 1), dtype=bool), ValueError, "at least one valid token"),
    ],
)
def test_generate_rejects_invalid_attention_mask_before_prefill(attention_mask, exception, match):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid generate inputs.")

    with pytest.raises(exception, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=attention_mask,
            kv_caches=(),
            call_hash="invalid-input",
            sampling_method=RngSamplingMethod(),
            max_new_tokens=1,
        )
