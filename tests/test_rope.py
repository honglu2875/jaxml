import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


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
