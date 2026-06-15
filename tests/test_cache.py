import numpy as np
import pytest
import jax.numpy as jnp

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


def test_kv_cache_rollback_rejects_past_beginning():
    k, v = _kv(seq_len=2)
    cache = KVCache.init(4).update(k, v, mask=jnp.ones((2, 2), dtype=bool))

    with pytest.raises(ValueError, match="past the beginning"):
        cache.rollback(3)


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
