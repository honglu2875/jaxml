import jax
import jax.numpy as jnp
import pytest

from jaxml.config import ModelConfig
from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.models.llama import LlamaModelWithHead

pytestmark = pytest.mark.tpu


def _tpu_devices():
    try:
        devices = jax.devices("tpu")
    except Exception as exc:
        pytest.skip(f"TPU backend is not available: {exc}")
    if not devices:
        pytest.skip("No TPU devices are visible to JAX.")
    return devices


def test_tpu_backend_smoke():
    devices = _tpu_devices()
    x = jax.device_put(jnp.arange(8, dtype=jnp.float32), devices[0])
    y = jax.jit(lambda z: (z * z).sum())(x)

    assert y == 140.0
    assert y.device.platform == "tpu"


def test_engine_reinit_params_on_tpu():
    devices = _tpu_devices()

    config = ModelConfig(
        hidden_size=32,
        head_dim=8,
        num_heads=4,
        num_layers=1,
        intermediate_ratio=(2, 1),
        max_position_embeddings=64,
        vocab_size=128,
        num_kv_heads=2,
        attn_scale=8**-0.5,
    )
    model = LlamaModelWithHead(config)
    tp_size = 1
    for candidate in (4, 2):
        if len(devices) % candidate == 0 and config.num_heads % candidate == 0 and config.num_key_value_heads % candidate == 0:
            tp_size = candidate
            break
    dp_size = len(devices) // tp_size

    engine = Engine(model, InferenceConfig(tp_size=tp_size, dp_size=dp_size), {})
    engine.init_params(use_tpu=True, reinit_weight=True)

    array_leaves = [leaf for leaf in jax.tree.leaves(engine.params) if hasattr(leaf, "addressable_shards")]
    assert array_leaves
    assert all(shard.device.platform == "tpu" for leaf in array_leaves for shard in leaf.addressable_shards)
