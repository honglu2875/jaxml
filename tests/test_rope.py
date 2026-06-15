import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.nn.position import RotaryEmbedding

pytestmark = pytest.mark.milestone


@pytest.mark.parametrize("model_type", ["llama", "neox"])
def test_rope(rope_factory, model_type):
    with jax.default_device(jax.devices("cpu")[0]):
        bs, seq = 4, 10
        hf_rope, rope, apply_fn = rope_factory(model_type)
        nh, hd = hf_rope.config.num_attention_heads, rope.dim
        key = jax.random.PRNGKey(0)
        q = jax.random.uniform(key, (bs, seq, nh, hd), dtype=jnp.float32)
        tq = torch.tensor(np.array(q))
        params = rope.init(key, q, seq_len=seq)

        cos, sin = rope.apply(params, q, seq_len=seq)
        cos_hf, sin_hf = hf_rope(tq, torch.arange(seq)[None])

        assert np.allclose(cos[:seq], cos_hf[0].numpy())
        assert np.allclose(sin[:seq], sin_hf[0].numpy())

        key = jax.random.PRNGKey(1)
        k = jax.random.uniform(key, (bs, seq, nh, hd), dtype=jnp.float32)
        tk = torch.tensor(np.array(k))

        oq, ok = rope.apply_rotary_pos_emb(q, k, cos, sin, jnp.arange(seq)[None])
        oq2, ok2 = apply_fn(tq.permute(0, 2, 1, 3), tk.permute(0, 2, 1, 3), cos_hf, sin_hf, rotary_ndims=cos.shape[-1])
        oq2 = oq2.permute(0, 2, 1, 3)
        ok2 = ok2.permute(0, 2, 1, 3)

        assert np.allclose(oq, oq2.numpy())
        assert np.allclose(ok, ok2.numpy())


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"dim": True, "max_length": 8}, TypeError, "dim must be an integer"),
        ({"dim": np.bool_(True), "max_length": 8}, TypeError, "dim must be an integer"),
        ({"dim": 1.5, "max_length": 8}, TypeError, "dim must be an integer"),
        ({"dim": 0, "max_length": 8}, ValueError, "dim must be positive"),
        ({"dim": 4, "max_length": True}, TypeError, "max_length must be an integer"),
        ({"dim": 4, "max_length": np.bool_(True)}, TypeError, "max_length must be an integer"),
        ({"dim": 4, "max_length": 1.5}, TypeError, "max_length must be an integer"),
        ({"dim": 4, "max_length": 0}, ValueError, "max_length must be positive"),
        ({"dim": 4, "max_length": 8, "base": True}, TypeError, "base must be a real number"),
        ({"dim": 4, "max_length": 8, "base": float("nan")}, ValueError, "base must be finite"),
        ({"dim": 4, "max_length": 8, "base": 0.0}, ValueError, "base must be positive"),
        ({"dim": 4, "max_length": 8, "rope_scale": "1.0"}, TypeError, "rope_scale must be a real number"),
        ({"dim": 4, "max_length": 8, "rope_scale": float("inf")}, ValueError, "rope_scale must be finite"),
        ({"dim": 4, "max_length": 8, "rope_scale": 0.5}, ValueError, "rope scale"),
        ({"dim": 4, "max_length": 8, "rotary_pct": np.bool_(True)}, TypeError, "rotary_pct must be a real number"),
        ({"dim": 4, "max_length": 8, "rotary_pct": float("nan")}, ValueError, "rotary_pct must be finite"),
        ({"dim": 4, "max_length": 8, "rotary_pct": 0.0}, ValueError, "rotary_pct must be in"),
        ({"dim": 4, "max_length": 8, "rotary_pct": 1.5}, ValueError, "rotary_pct must be in"),
        ({"dim": 4, "max_length": 8, "dtype": None}, TypeError, "dtype must be a valid JAX dtype"),
        ({"dim": 4, "max_length": 8, "dtype": "not-a-dtype"}, TypeError, "dtype must be a valid JAX dtype"),
        ({"dim": 4, "max_length": 8, "disable_cache": 1}, TypeError, "disable_cache must be a boolean"),
    ],
)
def test_rope_rejects_invalid_shape_parameters(kwargs, exception, match):
    rope = RotaryEmbedding(**kwargs)
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)

    with pytest.raises(exception, match=match):
        rope.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("disable_cache", [False, True])
def test_rope_accepts_numpy_scalar_parameters(disable_cache):
    rope = RotaryEmbedding(
        dim=np.int64(4),
        max_length=np.int64(8),
        base=np.float64(10000.0),
        dtype=np.float32,
        disable_cache=np.bool_(disable_cache),
        rotary_pct=np.float64(1.0),
        rope_scale=np.float64(1.0),
    )
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)

    params = rope.init(jax.random.PRNGKey(0), x, seq_len=np.int64(2))
    cos, sin = rope.apply(params, x, seq_len=np.int64(2))

    assert cos.dtype == jnp.float32
    assert sin.dtype == jnp.float32


@pytest.mark.parametrize("seq_len", [True, np.bool_(True), 1.5])
def test_rope_rejects_non_integer_seq_len(seq_len):
    rope = RotaryEmbedding(dim=4, max_length=8)
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)

    with pytest.raises(TypeError, match="seq_len must be an integer"):
        rope.init(jax.random.PRNGKey(0), x, seq_len=seq_len)


def test_cached_rope_rejects_seq_len_beyond_max_length():
    rope = RotaryEmbedding(dim=4, max_length=4)
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    params = rope.init(jax.random.PRNGKey(0), x, seq_len=2)

    with pytest.raises(ValueError, match="max_length=4"):
        rope.apply(params, x, seq_len=5)


def test_uncached_rope_can_compute_beyond_max_length():
    rope = RotaryEmbedding(dim=4, max_length=4, disable_cache=True)
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    params = rope.init(jax.random.PRNGKey(0), x, seq_len=2)

    cos, sin = rope.apply(params, x, seq_len=5)

    assert cos.shape == (5, 4)
    assert sin.shape == (5, 4)


@pytest.mark.parametrize(
    "x,exception,match",
    [
        (jnp.ones((2,), dtype=jnp.float32), ValueError, "batch and sequence axes"),
        (jnp.ones((1, 2, 1, 4), dtype=jnp.int32), TypeError, "x must contain floating point"),
    ],
)
def test_rope_rejects_invalid_input_tensor(x, exception, match):
    rope = RotaryEmbedding(dim=4, max_length=8)

    with pytest.raises(exception, match=match):
        rope.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        ({"q": jnp.ones((1, 2, 4)), "k": jnp.ones((1, 2, 1, 4))}, ValueError, "q must be a 4D array"),
        ({"q": jnp.ones((1, 2, 1, 4)), "k": jnp.ones((1, 2, 4))}, ValueError, "k must be a 4D array"),
        ({"q": jnp.ones((1, 2, 2, 4), dtype=jnp.int32)}, TypeError, "q must contain floating point"),
        ({"k": jnp.ones((1, 2, 1, 4), dtype=jnp.int32)}, TypeError, "k must contain floating point"),
        ({"cos": jnp.ones((4, 4), dtype=jnp.int32)}, TypeError, "cos must contain floating point"),
        ({"sin": jnp.ones((4, 4), dtype=jnp.int32)}, TypeError, "sin must contain floating point"),
        ({"q": jnp.ones((1, 3, 1, 4)), "k": jnp.ones((1, 2, 1, 4))}, ValueError, "matching batch, sequence"),
        ({"cos": jnp.ones((4,)), "sin": jnp.ones((4, 4))}, ValueError, "cos and sin must be 2D"),
        ({"cos": jnp.ones((4, 4)), "sin": jnp.ones((5, 4))}, ValueError, "same shape"),
        ({"cos": jnp.ones((0, 4)), "sin": jnp.ones((0, 4))}, ValueError, "at least one position"),
        ({"cos": jnp.ones((4, 3)), "sin": jnp.ones((4, 3))}, ValueError, "positive and even"),
        ({"cos": jnp.ones((4, 6)), "sin": jnp.ones((4, 6))}, ValueError, "cannot exceed"),
        ({"position_ids": jnp.arange(2)}, ValueError, "position_ids must be a 2D array"),
        ({"position_ids": jnp.arange(3, dtype=jnp.int32)[None]}, ValueError, "position_ids shape must be broadcastable"),
        ({"position_ids": jnp.arange(2, dtype=jnp.float32)[None]}, TypeError, "integer positions"),
        ({"position_ids": jnp.array([[-1, 0]], dtype=jnp.int32)}, ValueError, "non-negative positions"),
        ({"position_ids": jnp.array([[0, 4]], dtype=jnp.int32)}, ValueError, "within rotary table length"),
    ],
)
def test_apply_rotary_pos_emb_rejects_invalid_inputs(kwargs, exception, match):
    defaults = {
        "q": jnp.ones((1, 2, 2, 4), dtype=jnp.float32),
        "k": jnp.ones((1, 2, 1, 4), dtype=jnp.float32),
        "cos": jnp.ones((4, 4), dtype=jnp.float32),
        "sin": jnp.zeros((4, 4), dtype=jnp.float32),
        "position_ids": jnp.arange(2, dtype=jnp.int32)[None],
    }

    with pytest.raises(exception, match=match):
        RotaryEmbedding.apply_rotary_pos_emb(**(defaults | kwargs))


def test_apply_rotary_pos_emb_allows_grouped_query_head_counts():
    q = jnp.ones((1, 2, 2, 4), dtype=jnp.float32)
    k = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    cos = jnp.ones((4, 4), dtype=jnp.float32)
    sin = jnp.zeros((4, 4), dtype=jnp.float32)
    position_ids = jnp.arange(2, dtype=jnp.int32)[None]

    q_out, k_out = RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_apply_rotary_pos_emb_accepts_traced_position_ids():
    q = jnp.ones((1, 2, 2, 4), dtype=jnp.float32)
    k = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    cos = jnp.ones((4, 4), dtype=jnp.float32)
    sin = jnp.zeros((4, 4), dtype=jnp.float32)

    @jax.jit
    def apply(position_ids):
        return RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    q_out, k_out = apply(jnp.arange(2, dtype=jnp.int32)[None])

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_apply_rotary_pos_emb_masks_out_of_range_traced_position_ids():
    q = jnp.ones((1, 3, 2, 4), dtype=jnp.float32)
    k = jnp.ones((1, 3, 1, 4), dtype=jnp.float32)
    cos = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    sin = jnp.zeros((4, 4), dtype=jnp.float32)

    @jax.jit
    def apply(position_ids):
        return RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    q_out, k_out = apply(jnp.array([[0, -1, 4]], dtype=jnp.int32))

    assert np.all(np.isfinite(np.array(q_out)))
    assert np.all(np.isfinite(np.array(k_out)))
    assert np.allclose(
        q_out[:, 0], RotaryEmbedding.apply_rotary_pos_emb(q[:, :1], k[:, :1], cos, sin, jnp.array([[0]]))[0][:, 0]
    )
    assert np.array_equal(np.array(q_out[:, 1:]), np.zeros((1, 2, 2, 4), dtype=np.float32))
    assert np.array_equal(np.array(k_out[:, 1:]), np.zeros((1, 2, 1, 4), dtype=np.float32))
