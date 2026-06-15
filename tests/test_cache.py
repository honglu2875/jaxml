import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxml.cache import KVCache


def _kv(batch=2, seq_len=3, value=1.0):
    x = jnp.full((batch, seq_len, 1, 2), value, dtype=jnp.float32)
    return x, x + 1


def test_kv_cache_init_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="max_seq_len"):
        KVCache.init(0)


@pytest.mark.parametrize("max_seq_len", [True, np.bool_(True), 1.5])
def test_kv_cache_init_rejects_non_integer_capacity(max_seq_len):
    with pytest.raises(TypeError, match="max_seq_len must be an integer"):
        KVCache.init(max_seq_len)


def test_kv_cache_init_accepts_numpy_integer_capacity():
    cache = KVCache.init(np.int64(4))

    assert cache.max_seq_len == 4


@pytest.mark.parametrize("dtype", [None, "not-a-dtype", object()])
def test_kv_cache_init_rejects_invalid_dtype(dtype):
    with pytest.raises(TypeError, match="dtype must be a valid JAX dtype"):
        KVCache.init(4, dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, np.float32, "bfloat16"])
def test_kv_cache_init_canonicalizes_dtype(dtype):
    cache = KVCache.init(4, dtype=dtype)

    assert cache.dtype == jnp.dtype(dtype)


def test_kv_cache_init_accepts_complete_initial_state():
    k, v = _kv(seq_len=2)
    mask = jnp.array([[1, 1], [1, 0]], dtype=jnp.int32)

    cache = KVCache.init(4, k=k, v=v, mask=mask)

    assert cache.k.shape == (2, 4, 1, 2)
    assert cache.v.shape == (2, 4, 1, 2)
    assert cache.mask.dtype == jnp.bool_
    assert np.array_equal(np.array(cache.mask), np.array([[True, True, False, False], [True, False, False, False]]))
    assert np.array_equal(np.array(cache.pos_id), np.array([[1], [0]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": _kv(seq_len=1)[0]},
        {"v": _kv(seq_len=1)[1]},
        {"mask": jnp.ones((2, 1), dtype=bool)},
        {"k": _kv(seq_len=1)[0], "mask": jnp.ones((2, 1), dtype=bool)},
        {"v": _kv(seq_len=1)[1], "mask": jnp.ones((2, 1), dtype=bool)},
    ],
)
def test_kv_cache_init_rejects_partial_initial_state(kwargs):
    with pytest.raises(ValueError, match="requires both k and v"):
        KVCache.init(4, **kwargs)


def test_kv_cache_update_defaults_missing_initial_mask_to_valid_tokens():
    k, v = _kv(seq_len=2)

    cache = KVCache.init(4).update(k, v, mask=None)

    assert cache.k.shape == (2, 4, 1, 2)
    assert cache.v.shape == (2, 4, 1, 2)
    assert np.array_equal(np.array(cache.mask), np.array([[True, True, False, False], [True, True, False, False]]))
    assert np.array_equal(np.array(cache.pos_id), np.array([[1], [1]]))


def test_kv_cache_update_rejects_partial_empty_state():
    k, v = _kv(batch=1, seq_len=1)
    cache = KVCache(
        k=None,
        v=None,
        max_seq_len=4,
        mask=None,
        pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
    )

    with pytest.raises(ValueError, match="k is empty"):
        cache.update(k, v, mask=None)


@pytest.mark.parametrize(
    "k,v,match",
    [
        (_kv(seq_len=1)[0].astype(jnp.int32), _kv(seq_len=1)[1], "k must contain floating point"),
        (_kv(seq_len=1)[0], _kv(seq_len=1)[1].astype(jnp.int32), "v must contain floating point"),
    ],
)
def test_kv_cache_update_rejects_non_floating_values(k, v, match):
    with pytest.raises(TypeError, match=match):
        KVCache.init(4).update(k, v, mask=None)


def test_kv_cache_update_canonicalizes_integer_initial_mask():
    k, v = _kv(seq_len=3)
    mask = jnp.array([[1, 1, 0], [1, 0, 0]], dtype=jnp.int32)

    cache = KVCache.init(4).update(k, v, mask=mask)

    assert cache.mask.dtype == jnp.bool_
    assert np.array_equal(
        np.array(cache.mask),
        np.array([[True, True, False, False], [True, False, False, False]]),
    )
    assert np.array_equal(np.array(cache.pos_id), np.array([[1], [0]]))


def test_kv_cache_update_rejects_float_initial_mask():
    k, v = _kv(seq_len=3)
    mask = jnp.ones((2, 3), dtype=jnp.float32)

    with pytest.raises(TypeError, match="mask must be boolean or integer"):
        KVCache.init(4).update(k, v, mask=mask)


@pytest.mark.parametrize(
    "k,v",
    [
        (jnp.ones((2, 1), dtype=jnp.float32), jnp.ones((2, 1), dtype=jnp.float32)),
        (jnp.ones((2, 1, 2), dtype=jnp.float32), jnp.ones((2, 1, 2), dtype=jnp.float32)),
    ],
)
def test_kv_cache_update_rejects_non_4d_key_values(k, v):
    with pytest.raises(ValueError, match="4D arrays"):
        KVCache.init(4).update(k, v, mask=None)


def test_kv_cache_next_pos_id_rejects_uninitialized_cache():
    with pytest.raises(ValueError, match="before KV cache initialization"):
        _ = KVCache.init(4).next_pos_id


def test_kv_cache_update_rejects_prompt_longer_than_capacity():
    k, v = _kv(seq_len=3)

    with pytest.raises(ValueError, match="Cannot cache 3 tokens"):
        KVCache.init(2).update(k, v, mask=jnp.ones((2, 3), dtype=bool))


def test_kv_cache_update_rejects_initial_mask_without_valid_tokens():
    k, v = _kv(seq_len=3)
    mask = jnp.array([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="at least one valid token"):
        KVCache.init(4).update(k, v, mask=mask)


def test_kv_cache_update_rejects_invalid_decode_length():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 2), dtype=bool))

    with pytest.raises(ValueError, match="exactly one token"):
        cache.update(k, v, mask=None)


def test_kv_cache_update_rejects_decode_past_capacity():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(2).update(k, v, mask=jnp.ones((2, 2), dtype=bool))
    next_k, next_v = _kv(seq_len=1)

    with pytest.raises(ValueError, match="max_seq_len=2"):
        cache.update(next_k, next_v, mask=None)


def test_kv_cache_update_rejects_decode_mask():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 2), dtype=bool))
    next_k, next_v = _kv(seq_len=1)

    with pytest.raises(ValueError, match="Decode cache mask shape must match"):
        cache.update(next_k, next_v, mask=jnp.ones((2, 1), dtype=bool))


def test_kv_cache_update_rejects_stale_decode_mask():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.array([[True, True], [True, False]]))
    next_k, next_v = _kv(seq_len=1, value=3.0)
    stale_mask = jnp.ones((2, 4), dtype=bool)

    with pytest.raises(ValueError, match="must match current cached mask state"):
        cache.update(next_k, next_v, mask=stale_mask)


def test_kv_cache_update_rejects_populated_cache_mask_without_valid_tokens():
    k, v = _kv(batch=1, seq_len=1)
    cache = KVCache(
        k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
        v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
        max_seq_len=4,
        mask=jnp.zeros((1, 4), dtype=bool),
        pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
    )

    with pytest.raises(ValueError, match="Cached mask must contain at least one valid token"):
        cache.update(k, v, mask=None)


@pytest.mark.parametrize(
    "cache,exception,match",
    [
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 3, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            ValueError,
            "same shape",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            ValueError,
            "4D arrays",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 3, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 3, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 3), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            ValueError,
            "sequence axis must match",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.int32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            TypeError,
            "Cached k must contain floating",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.int32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            TypeError,
            "Cached v must contain floating",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 3), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            ValueError,
            "Cached mask shape",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=jnp.float32),
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            TypeError,
            "Cached mask must be boolean or integer",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.zeros((1,), dtype=jnp.int32),
            ),
            ValueError,
            "Cached pos_id shape",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.zeros((1, 1), dtype=jnp.float32),
            ),
            TypeError,
            "Cached pos_id must contain integer",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=jnp.ones((1, 4), dtype=bool),
                pos_id=jnp.array([[4]], dtype=jnp.int32),
            ),
            ValueError,
            r"within \[0, 4\)",
        ),
    ],
)
def test_kv_cache_update_rejects_invalid_populated_state(cache, exception, match):
    k, v = _kv(batch=1, seq_len=1)

    with pytest.raises(exception, match=match):
        cache.update(k, v, mask=None)


@pytest.mark.parametrize("mask_arg", ["none", "cached"])
def test_kv_cache_update_decode_extends_cached_mask(mask_arg):
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.array([[True, True], [True, False]]))
    next_k, next_v = _kv(seq_len=1, value=3.0)
    mask = None if mask_arg == "none" else cache.mask

    cache = cache.update(next_k, next_v, mask=mask)

    assert np.array_equal(
        np.array(cache.mask),
        np.array([[True, True, True, False], [True, True, False, False]]),
    )
    assert np.array_equal(np.array(cache.pos_id), np.array([[2], [1]]))


def test_kv_cache_rollback_rejects_past_beginning():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 2), dtype=bool))

    with pytest.raises(ValueError, match="past the beginning"):
        cache.rollback(3)


def test_kv_cache_rollback_accepts_traced_cache_state():
    k, v = _kv(seq_len=3)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 3), dtype=bool))

    @jax.jit
    def rollback(cache):
        return cache.rollback(1)

    rolled_back = rollback(cache)

    assert np.array_equal(np.array(rolled_back.pos_id), np.array([[1], [1]]))
    assert np.array_equal(
        np.array(rolled_back.mask),
        np.array([[True, True, False, False], [True, True, False, False]]),
    )
    assert np.array_equal(np.array(rolled_back.k[:, 2:]), np.zeros((2, 2, 1, 2), dtype=np.float32))


@pytest.mark.parametrize("method_name", ["rollback", "resize"])
@pytest.mark.parametrize(
    "cache,match",
    [
        (
            KVCache(
                k=None,
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=None,
                pos_id=None,
            ),
            "k is empty",
        ),
        (
            KVCache(
                k=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                v=jnp.zeros((1, 4, 1, 2), dtype=jnp.float32),
                max_seq_len=4,
                mask=None,
                pos_id=jnp.zeros((1, 1), dtype=jnp.int32),
            ),
            "populated k requires",
        ),
    ],
)
def test_kv_cache_mutations_reject_partial_state(method_name, cache, match):
    method = getattr(cache, method_name)
    args = (1,) if method_name == "rollback" else (8,)

    with pytest.raises(ValueError, match=match):
        method(*args)


@pytest.mark.parametrize("n", [True, np.bool_(True), 1.5])
def test_kv_cache_rollback_rejects_non_integer_count(n):
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 2), dtype=bool))

    with pytest.raises(TypeError, match="n must be an integer"):
        cache.rollback(n)


def test_kv_cache_resize_rejects_truncating_cached_positions():
    k, v = _kv(seq_len=3)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 3), dtype=bool))

    with pytest.raises(ValueError, match="below the highest cached position"):
        cache.resize(2)


def test_kv_cache_resize_accepts_traced_cache_state():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 2), dtype=bool))

    @jax.jit
    def resize(cache):
        return cache.resize(3)

    resized = resize(cache)

    assert resized.max_seq_len == 3
    assert resized.k.shape == (2, 3, 1, 2)
    assert resized.v.shape == (2, 3, 1, 2)
    assert np.array_equal(np.array(resized.pos_id), np.array([[1], [1]]))
    assert np.array_equal(
        np.array(resized.mask),
        np.array([[True, True, False], [True, True, False]]),
    )


@pytest.mark.parametrize("new_size", [True, np.bool_(True), 1.5])
def test_kv_cache_resize_rejects_non_integer_size(new_size):
    with pytest.raises(TypeError, match="new_size must be an integer"):
        KVCache.init(4).resize(new_size)


def test_kv_cache_resize_accepts_numpy_integer_size():
    cache = KVCache.init(4).resize(np.int64(8))

    assert cache.max_seq_len == 8


def test_kv_cache_resize_empty_cache_updates_capacity():
    cache = KVCache.init(4).resize(8)

    assert cache.max_seq_len == 8
    assert cache.k is None
    assert cache.v is None
    assert cache.mask is None
