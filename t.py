from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import make_causal_mask, make_local_attention_mask
from jax import random
import jax.numpy as jnp

causal = make_causal_mask(shape=(100, 100))
local = make_local_attention_mask(shape=(100, 100), window_size=20)

mask = causal | local

num_heads = 2
dtype=jnp.bfloat16
k1, k2, k3 = random.split(random.key(0), 3)
seq_len = 1024
head_dim = 128
q = random.uniform(k1, (num_heads, seq_len, head_dim), dtype=dtype)
k = random.uniform(k2, (num_heads, seq_len, head_dim), dtype=dtype)
v = random.uniform(k3, (num_heads, seq_len, head_dim), dtype=dtype)

