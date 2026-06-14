import jax.numpy as jnp

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
