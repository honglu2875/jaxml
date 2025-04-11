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
from jaxml.inference_engine.sampling import SamplingMethod
from jaxml.outputs import GenerationOutput
from jaxml.utils import _hash, timeit

logger = logging.getLogger(__name__)


@struct.dataclass
class InferenceConfig:
    tp_size: int = 1
    dp_size: int = 1


CACHE_STRIDE = 256


class Engine:
    """Wrap around a model class to do autoregressive generation."""

    def __init__(self, model: nn.Module, config: InferenceConfig, params: FrozenDict, dtype: Any = jnp.float32):
        self.model = model
        assert config.tp_size * config.dp_size <= jax.device_count()
        self.config = config
        self.params = params
        self.dtype = dtype

    def init_cache(self, max_seq_len: int) -> list[KVCache]:
        max_seq_len = max_seq_len
        num_layers = self.model.config.num_layers
        return tuple(KVCache.init(max_seq_len, None, None, dtype=self.dtype) for _ in range(num_layers))

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
            ("embed", None),
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
        seed: int = 0,
        max_new_tokens: int = 10,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        fuse_decoding: bool = False,
        include_prompt: bool = True,
    ):
        apply = self.wrapped_apply_fn

        from .._generate import generate

        sampling_method = SamplingMethod.from_values(top_k=top_k, top_p=top_p, min_p=min_p, temp=temperature)
        logger.info(
            f"Given the parameters {top_k=}, {top_p=}, {min_p=}, {temperature=}, the sampling method is determined as follows: {str(sampling_method)}."
        )

        # For every unique model call, cache the AOT-compiled function to disk
        # Note: top_k value cannot be traced and need to be hashed as well.
        top_k = 0 if top_k < 0 else top_k
        prompt_len = prompt_tokens.shape[1]
        total_len = prompt_len + max_new_tokens

        output_tokens = [prompt_tokens] if include_prompt else []
        initial_buffer_len = (prompt_len // CACHE_STRIDE + 1) * CACHE_STRIDE
        kv_caches = self.init_cache(max_seq_len=initial_buffer_len)
        for cache_len in range(initial_buffer_len, total_len + CACHE_STRIDE, CACHE_STRIDE):
            # Every step, the kv-cache max length (`cache_len`) is set up to be a multiple of CACHE_STRIDE
            # `new_token_count` keeps track of the remaining tokens to fill
            new_token_count = min(cache_len - prompt_len, CACHE_STRIDE, CACHE_STRIDE - cache_len + total_len)

            if cache_len > initial_buffer_len:
                kv_caches = tuple(c.resize(cache_len) for c in kv_caches)

            call_hash = _hash(
                str(self.model),
                str(self.config),
                str(prompt_tokens.shape),
                str(sampling_method),
                str(top_k),
                str(cache_len),
            )

            step_output: GenerationOutput = generate(
                self.params,
                apply,
                prompt_tokens if not output_tokens else output_tokens[-1],
                attention_mask,
                kv_caches,
                call_hash,
                sampling_method,
                seed=seed,
                max_new_tokens=new_token_count,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
                fuse_decoding=fuse_decoding,
                include_prompt=False,
                skip_prefill=(cache_len > initial_buffer_len),
            )
            output_tokens.append(step_output.tokens)
            kv_caches = step_output.kv_caches

        return jnp.concatenate(output_tokens, axis=-1)
