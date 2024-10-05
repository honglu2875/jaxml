import logging
import warnings
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import FrozenDict
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxml.cache import KVCache
from jaxml.utils import timeit

logger = logging.getLogger(__name__)


@struct.dataclass
class InferenceConfig:
    tp_size: int = 1
    dp_size: int = 1
    max_sequence_length: int = 128
    # JIT-compile configs
    length_stride: Optional[int] = None


class Engine:
    """Wrap around a model class to do autoregressive generation."""

    model: nn.Module
    config: InferenceConfig
    params: FrozenDict
    kv_cache: Optional[list[KVCache]]
    dtype: Any = jnp.bfloat16

    def __init__(self, model: nn.Module, config: InferenceConfig, params: FrozenDict):
        self.model = model
        assert config.tp_size * config.dp_size <= jax.device_count()
        self.config = config
        self.params = params

    def init_cache(self, max_seq_len: Optional[int] = None) -> list[KVCache]:
        max_seq_len = max_seq_len or self.config.max_sequence_length
        num_layers = self.model.config.num_layers
        return [KVCache.init(max_seq_len, None, None, dtype=self.dtype) for _ in range(num_layers)]

    @staticmethod
    def mesh_sharding(pspec: Optional[PartitionSpec], mesh: Optional[Mesh]) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    def _shard_params(self, x: Any, y: PartitionSpec):
        if x.ndim != len(y.spec):
            assert (
                x.ndim == 2 and len(y.spec) == 3
            ), f"The shape of x ({x.shape}) and the sharding spec ({y.spec}) does not match"
            warnings.warn(
                f"The parameter has 2 axis ({x.shape}) while the sharding spec ({y.spec}) has 3 axis. "
                "Attempting to reshape into [:, :, head_dim], but please confirm that this is the intended behavior."
            )
            return jax.device_put(
                x.reshape(
                    (
                        x.shape[0],
                        -1,
                        self.head_dim,
                    )
                ),
                y,
            )
        return jax.device_put(x, y)

    @timeit(logger)
    def init_params(self, weights: Optional = None, use_tpu: bool = False, reinit_weight: bool = False):
        """
        Re-initialize the properly sharded parameters.

        Args:
            weights: whether to provide a set of custom model weights. If None, defaults to self.params.
            use_tpu: whether to use tpu instead of cpu.
            reinit_weight: if True, it will ignore `weights` and re-init the model parameters.
        Returns:
            a tree of properly sharded parameters
        """
        tp_size = self.config.tp_size
        dp_size = self.config.dp_size
        weights = weights or self.params
        # whether: use 1 single TPU or CPU
        is_single_device = (tp_size * dp_size == 1) or not use_tpu

        key = jax.random.PRNGKey(0)

        # (dp, tp)
        mesh_layout = (dp_size, tp_size)

        dummy_input = jnp.array([[1 for _ in range(mesh_layout[1])] for _ in range(mesh_layout[0])])
        abstract_variables = jax.eval_shape(self.model.init, key, dummy_input)

        # The sharding rules mapping named logical axis to named axis
        # Changing sharding configuration requires changing this tuple.
        rules = (
            ("batch", "data"),
            ("heads", "model"),
            ("kv_length", None),
            ("length", None),
            ("intermediate", "model"),
            ("heads_merged", "model"),
            ("head_states", None),
        )

        if not is_single_device:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(mesh_layout),
                axis_names=("data", "model"),
            )

            logical_state_spec = nn.get_partition_spec(abstract_variables)
            logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, rules)

            input_sharding = self.mesh_sharding(PartitionSpec("data", None), mesh)

        # In case sharded==False, use _single_device_fn to move devices accordingly
        _single_device_fn = jnp.array if use_tpu else np.array

        if reinit_weight:
            if not is_single_device:
                # Directly init to sharded devices
                params = jax.jit(
                    self.model.init,
                    in_shardings=(
                        self.mesh_sharding(None, mesh),
                        input_sharding,
                    ),  # PRNG key and x
                    out_shardings=logical_state_sharding,
                )(key, dummy_input)
                self.params = params
                return
            else:
                # Init weights on CPU first
                weights = self.model.init(key, dummy_input)

        # Can assume weight is not None from now, and the goal is only to shard it
        assert isinstance(weights, dict), f"weights must be a dict, got {type(weights)}"
        assert "params" in weights, f"The key params not found in 'weights'. Got {weights.keys()}"
        if not is_single_device:
            params = {
                "params": jax.tree.map(
                    self._shard_params,
                    weights["params"],
                    logical_state_sharding["params"],
                ),
                **{k: jax.tree.map(_single_device_fn, v) for k, v in weights.items() if k != "params"},
            }
        else:
            params = jax.tree.map(_single_device_fn, weights)

        self.params = params

    def prepare_input(self, inputs, dtype: Any = None):
        tp_size = self.config.tp_size
        dp_size = self.config.dp_size

        mesh = Mesh(
            devices=mesh_utils.create_device_mesh((dp_size, tp_size)),
            axis_names=("data", "model"),
        )
        inputs = jax.device_put(inputs, self.mesh_sharding(PartitionSpec("data", None), mesh))
        if dtype is not None:
            inputs = jax.tree.map(lambda x: x.astype(dtype), inputs)
        return inputs

    def wrapped_apply_fn(
        self,
        params,
        input_token,
        attention_mask=None,
        kv_caches=None,
        use_cache=True,
    ) -> tuple[jnp.ndarray, list[KVCache]]:

        out, _ = self.model.apply(
            params,
            input_token,
            position_ids=None,
            attention_mask=attention_mask,
            mutable=("cache",),
            # output_hidden_states=False, # maybe allow for toggling of hidden states in the future
            # output_attentions=False, # maybe allow for toggling of attn wts in the future
            kv_caches=kv_caches,
            use_cache=use_cache,
        )  # return a tuple (CausalLMOutputWithCache, dict) where dict is the mutable cache

        return out.logits, out.kv_caches

    def generate(
        self,
        prompt_tokens: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        do_sample: bool = True,
        seed: int = 0,
        max_new_tokens: int = 10,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        no_jit: bool = False,
        show_progress: bool = False,
    ):
        if no_jit:
            apply = self.wrapped_apply_fn
        else:
            apply = jax.jit(self.wrapped_apply_fn, static_argnames=("use_cache",))

        kv_caches = self.init_cache(max_seq_len=prompt_tokens.shape[1] + max_new_tokens)

        from .._generate import generate

        return generate(
            self.params,
            apply,
            prompt_tokens,
            attention_mask,
            kv_caches,
            do_sample=do_sample,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            show_progress=show_progress,
        )
