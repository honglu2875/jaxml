import jax
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


class RecordingSamplingMethod:
    def __init__(self):
        self.calls = []

    def get_sampling_fn(self):
        def sample_fn(rng, logits, top_k, top_p, min_p, temp):
            del rng
            self.calls.append((top_k, top_p, min_p, temp))
            return jnp.zeros(logits.shape[:-1], dtype=jnp.int32)

        return sample_fn


class MissingGetSamplingFn:
    pass


class NonCallableSamplingFn:
    def get_sampling_fn(self):
        return None


class StaticSamplingMethod:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def get_sampling_fn(self):
        def sample_fn(rng, logits, top_k, top_p, min_p, temp):
            del rng, logits, top_k, top_p, min_p, temp
            return self.token_ids

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


def test_generate_rejects_non_callable_eval_fn_before_prefill():
    with pytest.raises(TypeError, match="eval_fn must be callable"):
        generate(
            {},
            eval_fn=None,
            prompt_tokens=jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-eval-fn",
            sampling_method=RngSamplingMethod(),
            max_new_tokens=1,
        )


@pytest.mark.parametrize(
    "sampling_method,match",
    [
        (MissingGetSamplingFn(), "callable get_sampling_fn"),
        (NonCallableSamplingFn(), "must return a callable"),
    ],
)
def test_generate_rejects_invalid_sampling_method_before_prefill(sampling_method, match):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid sampling methods.")

    with pytest.raises(TypeError, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-sampling-method",
            sampling_method=sampling_method,
            max_new_tokens=1,
        )


@pytest.mark.parametrize(
    "eval_output,exception,match",
    [
        (object(), TypeError, "pair of"),
        ((jnp.ones((1, 1), dtype=jnp.float32), ()), ValueError, "input_tokens.shape"),
        ((jnp.ones((2, 1, 4), dtype=jnp.float32), ()), ValueError, "leading shape"),
        ((jnp.ones((1, 1, 0), dtype=jnp.float32), ()), ValueError, "non-empty vocabulary"),
        ((jnp.ones((1, 1, 4), dtype=jnp.int32), ()), TypeError, "floating dtype"),
    ],
)
def test_generate_rejects_invalid_eval_outputs(monkeypatch, eval_output, exception, match):
    def fake_load_if_exists(name, hash, log=True):
        del name, hash, log

        def decorator(fn):
            return fn

        return decorator

    def eval_fn(params, tokens, attention_mask=None, kv_caches=None, use_cache=True):
        del params, tokens, attention_mask, kv_caches, use_cache
        return eval_output

    monkeypatch.setattr("jaxml._generate.load_if_exists", fake_load_if_exists)

    with pytest.raises(exception, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-eval-output",
            sampling_method=RngSamplingMethod(),
            max_new_tokens=1,
        )


@pytest.mark.parametrize(
    "token_ids,exception,match",
    [
        (jnp.zeros((1,), dtype=jnp.int32), ValueError, "shape"),
        (jnp.zeros((1, 1), dtype=jnp.float32), TypeError, "integer token ids"),
    ],
)
def test_generate_rejects_invalid_sampled_tokens(monkeypatch, token_ids, exception, match):
    def fake_load_if_exists(name, hash, log=True):
        del name, hash, log

        def decorator(fn):
            return fn

        return decorator

    def eval_fn(params, tokens, attention_mask=None, kv_caches=None, use_cache=True):
        del params, attention_mask, use_cache
        logits = jnp.zeros(tokens.shape + (10,), dtype=jnp.float32)
        return logits, kv_caches

    monkeypatch.setattr("jaxml._generate.load_if_exists", fake_load_if_exists)

    with pytest.raises(exception, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-sampled-tokens",
            sampling_method=StaticSamplingMethod(token_ids),
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


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"top_k": True}, TypeError, "top_k must be an integer"),
        ({"top_k": 1.5}, TypeError, "top_k must be an integer"),
        ({"top_p": True}, TypeError, "top_p must be a real number"),
        ({"min_p": "0.1"}, TypeError, "min_p must be a real number"),
        ({"temperature": np.nan}, ValueError, "temp must be finite"),
    ],
)
def test_generate_rejects_invalid_sampling_arguments_before_prefill(kwargs, exception, match):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid generate inputs.")

    with pytest.raises(exception, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-sampling-input",
            sampling_method=RngSamplingMethod(),
            max_new_tokens=1,
            **kwargs,
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


def test_generate_forwards_clipped_sampling_arguments_to_sampling_method(monkeypatch):
    def fake_load_if_exists(name, hash, log=True):
        del name, hash, log

        def decorator(fn):
            return fn

        return decorator

    def eval_fn(params, tokens, attention_mask=None, kv_caches=None, use_cache=True):
        del params, attention_mask, use_cache
        logits = jnp.zeros(tokens.shape + (10,), dtype=jnp.float32)
        return logits, kv_caches

    monkeypatch.setattr("jaxml._generate.load_if_exists", fake_load_if_exists)
    sampling_method = RecordingSamplingMethod()

    output = generate(
        {},
        eval_fn,
        jnp.ones((1, 1), dtype=jnp.int32),
        attention_mask=None,
        kv_caches=(),
        call_hash="sampling-clip",
        sampling_method=sampling_method,
        max_new_tokens=1,
        top_k=-1,
        top_p=2.0,
        min_p=-0.5,
        temperature=-1.0,
    )

    assert output.tokens.shape == (1, 1)
    assert sampling_method.calls == [(0, 1.0, 0.0, 0.0)]


@pytest.mark.parametrize(
    "include_prompt,expected_tokens",
    [
        (False, np.empty((1, 0), dtype=np.int32)),
        (True, np.array([[3, 4]], dtype=np.int32)),
    ],
)
def test_generate_zero_new_tokens_returns_without_prefill(include_prompt, expected_tokens):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called when max_new_tokens is zero.")

    rng = jnp.array([0, 9], dtype=jnp.uint32)
    output = generate(
        {},
        eval_fn,
        jnp.array([[3, 4]], dtype=jnp.int32),
        attention_mask=jnp.array([[1, 1]], dtype=jnp.int32),
        kv_caches=(),
        call_hash="zero-new-tokens",
        sampling_method=RngSamplingMethod(),
        rng=rng,
        max_new_tokens=0,
        include_prompt=include_prompt,
    )

    assert np.array_equal(np.array(output.tokens), expected_tokens)
    assert np.array_equal(np.array(output.rng), np.array(rng))


def test_generate_returns_rng_for_decoding_continuation(monkeypatch):
    def fake_load_if_exists(name, hash, log=True):
        del name, hash, log

        def decorator(fn):
            return fn

        return decorator

    def eval_fn(params, tokens, attention_mask=None, kv_caches=None, use_cache=True):
        del params, attention_mask, use_cache
        logits = jnp.zeros(tokens.shape + (10,), dtype=jnp.float32)
        return logits, kv_caches

    monkeypatch.setattr("jaxml._generate.load_if_exists", fake_load_if_exists)
    initial_rng = jnp.array([0, 7], dtype=jnp.uint32)

    output = generate(
        {},
        eval_fn,
        jnp.ones((1, 1), dtype=jnp.int32),
        attention_mask=None,
        kv_caches=(),
        call_hash="rng-continuation",
        sampling_method=RngSamplingMethod(),
        rng=initial_rng,
        max_new_tokens=3,
    )

    expected_rng = initial_rng
    for _ in range(2):
        expected_rng, _ = jax.random.split(expected_rng)

    assert output.tokens.shape == (1, 3)
    assert np.array_equal(np.array(output.rng), np.array(expected_rng))


@pytest.mark.parametrize(
    "rng,exception,match",
    [
        (jnp.ones((1, 2), dtype=jnp.uint32), ValueError, "shape"),
        (jnp.ones((2,), dtype=jnp.float32), TypeError, "integer key data"),
    ],
)
def test_generate_rejects_invalid_rng_before_prefill(rng, exception, match):
    def eval_fn(*args, **kwargs):
        raise AssertionError("eval_fn should not be called for invalid generate inputs.")

    with pytest.raises(exception, match=match):
        generate(
            {},
            eval_fn,
            jnp.ones((1, 1), dtype=jnp.int32),
            attention_mask=None,
            kv_caches=(),
            call_hash="invalid-input",
            sampling_method=RngSamplingMethod(),
            rng=rng,
            max_new_tokens=1,
        )


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
