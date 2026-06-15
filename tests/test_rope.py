import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from jaxml.nn.position import RotaryEmbedding


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
        ({"dim": 1.5, "max_length": 8}, TypeError, "dim must be an integer"),
        ({"dim": 0, "max_length": 8}, ValueError, "dim must be positive"),
        ({"dim": 4, "max_length": True}, TypeError, "max_length must be an integer"),
        ({"dim": 4, "max_length": 1.5}, TypeError, "max_length must be an integer"),
        ({"dim": 4, "max_length": 0}, ValueError, "max_length must be positive"),
    ],
)
def test_rope_rejects_invalid_shape_parameters(kwargs, exception, match):
    rope = RotaryEmbedding(**kwargs)
    x = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)

    with pytest.raises(exception, match=match):
        rope.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize("seq_len", [True, 1.5])
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
    "kwargs,exception,match",
    [
        ({"q": jnp.ones((1, 2, 4)), "k": jnp.ones((1, 2, 1, 4))}, ValueError, "q must be a 4D array"),
        ({"q": jnp.ones((1, 2, 1, 4)), "k": jnp.ones((1, 2, 4))}, ValueError, "k must be a 4D array"),
        ({"q": jnp.ones((1, 3, 1, 4)), "k": jnp.ones((1, 2, 1, 4))}, ValueError, "matching batch, sequence"),
        ({"cos": jnp.ones((4,)), "sin": jnp.ones((4, 4))}, ValueError, "cos and sin must be 2D"),
        ({"cos": jnp.ones((4, 4)), "sin": jnp.ones((5, 4))}, ValueError, "same shape"),
        ({"cos": jnp.ones((4, 3)), "sin": jnp.ones((4, 3))}, ValueError, "positive and even"),
        ({"cos": jnp.ones((4, 6)), "sin": jnp.ones((4, 6))}, ValueError, "cannot exceed"),
        ({"position_ids": jnp.arange(2)}, ValueError, "position_ids must be a 2D array"),
        ({"position_ids": jnp.arange(3, dtype=jnp.int32)[None]}, ValueError, "position_ids shape must be broadcastable"),
        ({"position_ids": jnp.arange(2, dtype=jnp.float32)[None]}, TypeError, "integer positions"),
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
