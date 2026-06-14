import logging
import warnings
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core import FrozenDict
from flax.linen import logical_to_mesh_sharding
from flax.typing import FrozenVariableDict
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxml.cache import KVCache
from jaxml.inference_engine.sampling import SamplingMethod
from jaxml.outputs import GenerationOutput
from jaxml.utils import _hash, timeit

logger = logging.getLogger(__name__)

GENERATION_AOT_CACHE_VERSION = "generate_v2"


@struct.dataclass
class InferenceConfig:
    tp_size: int = 1
    dp_size: int = 1

    def __post_init__(self):
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}.")
        if self.dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {self.dp_size}.")


class Engine:
    """Wrap around a model class to do autoregressive generation."""

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        params: FrozenDict,
        dtype: Any = jnp.float32,
        cache_stride: int = 256,
    ):
        if cache_stride <= 0:
            raise ValueError(f"cache_stride must be positive, got {cache_stride}.")
        required_devices = config.tp_size * config.dp_size
        available_devices = jax.device_count()
        if required_devices > available_devices:
            raise ValueError(
                f"InferenceConfig requires {required_devices} devices "
                f"(tp_size={config.tp_size}, dp_size={config.dp_size}), but only {available_devices} are available."
            )
        self.model = model
        self.config = config
        self.params = params
        self.dtype = dtype
        self.cache_stride = cache_stride

    def init_cache(self, max_seq_len: int) -> tuple[KVCache, ...]:
        max_seq_len = max_seq_len
        num_layers = self.model.config.num_layers
        return tuple(KVCache.init(max_seq_len, None, None, dtype=self.dtype) for _ in range(num_layers))

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec, mesh: Optional[Mesh]) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    def _shard_params(self, x: Any, y: PartitionSpec):
        assert hasattr(y, "spec")
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
                        self.model.head_dim,
                    )
                ),
                y,
            )
        return jax.device_put(x, y)

    @staticmethod
    def _pytree_signature(tree: Any) -> str:
        def _leaf_signature(x):
            shape = getattr(x, "shape", None)
            dtype = getattr(x, "dtype", None)
            return (type(x).__module__, type(x).__qualname__, shape, str(dtype))

        leaves = jax.tree.leaves(tree)
        return str((jax.tree.structure(tree), tuple(_leaf_signature(x) for x in leaves)))

    @staticmethod
    def _platform_signature(tree: Any) -> str:
        import jaxlib

        platforms = set()
        for leaf in jax.tree.leaves(tree):
            device = getattr(leaf, "device", None)
            if device is not None:
                platforms.add(getattr(device, "platform", str(device)))
            for shard in getattr(leaf, "addressable_shards", ()):
                platforms.add(shard.device.platform)
        if not platforms:
            platforms.add(jax.default_backend())
        return str((jax.__version__, jaxlib.__version__, jax.default_backend(), tuple(sorted(platforms))))

    @timeit(logger)
    def init_params(self, weights: Optional[FrozenDict] = None, use_tpu: bool = False, reinit_weight: bool = False):
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
            logical_state_sharding = logical_to_mesh_sharding(logical_state_spec, mesh, rules)
            input_sharding = self.mesh_sharding(PartitionSpec("data", None), mesh)
            params_fn = jax.jit(
                self.model.init,
                in_shardings=(
                    self.mesh_sharding(PartitionSpec(), mesh),
                    input_sharding,
                ),  # PRNG key and x
                out_shardings=logical_state_sharding,
            )
        else:
            def params_fn(*_):
                return FrozenDict({})

            logical_state_sharding = {"params": PartitionSpec()}

        # In case sharded==False, use _single_device_fn to move devices accordingly
        _single_device_fn = jnp.array if use_tpu else np.array

        if reinit_weight:
            if not is_single_device:
                # Directly init to sharded devices
                self.params = params_fn(key, dummy_input)
                return
            else:
                # Init weights on CPU first
                weights: FrozenVariableDict | dict = self.model.init(key, dummy_input)

        # Can assume weight is not None from now, and the goal is only to shard it
        assert isinstance(weights, dict), f"weights must be a dict, got {type(weights)}"
        assert "params" in weights, f"The key params not found in 'weights'. Got {weights.keys()}"
        if not is_single_device:
            params = FrozenDict(
                {
                    "params": jax.tree.map(
                        self._shard_params,
                        weights["params"],
                        logical_state_sharding["params"],
                    ),
                    **{k: jax.tree.map(_single_device_fn, v) for k, v in weights.items() if k != "params"},
                }
            )
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
    ) -> tuple[jnp.ndarray, tuple[KVCache, ...]]:

        out, _ = self.model.apply(
            params,
            input_token,
            position_ids=None,
            attention_mask=attention_mask,
            mutable=("cache",),
            # output_hidden_states=False, # TODO: maybe allow for toggling of hidden states in the future
            # output_attentions=False, # TODO: maybe allow for toggling of attn wts in the future
            kv_caches=kv_caches,
            use_cache=use_cache,
            keep_last_n_logits=1,
        )  # return a tuple (CausalLMOutputWithCache, dict) where dict is the mutable cache

        return out.logits, out.kv_caches

    @staticmethod
    def _prepare_generation_inputs(
        prompt_tokens: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        prompt_tokens = jnp.asarray(prompt_tokens)
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None]
        elif prompt_tokens.ndim != 2:
            raise ValueError(f"prompt_tokens must be a 1D or 2D array, got shape {prompt_tokens.shape}.")

        if prompt_tokens.shape[1] == 0:
            raise ValueError("prompt_tokens must contain at least one token.")

        if attention_mask is None:
            return prompt_tokens, None

        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None]
        elif attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be a 1D or 2D array, got shape {attention_mask.shape}.")

        if attention_mask.shape != prompt_tokens.shape:
            raise ValueError(
                f"attention_mask shape must match prompt_tokens shape; got {attention_mask.shape} and {prompt_tokens.shape}."
            )

        return prompt_tokens, attention_mask

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
        prompt_tokens, attention_mask = self._prepare_generation_inputs(prompt_tokens, attention_mask)
        apply = self.wrapped_apply_fn

        from .._generate import generate

        sampling_method = SamplingMethod.from_values(top_k=top_k, top_p=top_p, min_p=min_p, temp=temperature)
        logger.info(
            "Given the parameters top_k=%.2f, top_p=%.2f, min_p=%.2f, temperature=%.2f, "
            "the sampling method is determined as follows: %s.",
            top_k,
            top_p,
            min_p,
            temperature,
            str(sampling_method),
        )

        # For every unique model call, cache the AOT-compiled function to disk
        # Note: top_k value cannot be traced and need to be hashed as well.
        top_k = 0 if top_k < 0 else top_k
        prompt_len = prompt_tokens.shape[1]

        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
        if max_new_tokens == 0:
            if include_prompt:
                return prompt_tokens
            return jnp.empty((prompt_tokens.shape[0], 0), dtype=prompt_tokens.dtype)

        output_tokens = []
        next_input_tokens = prompt_tokens
        generated_count = 0
        initial_buffer_len = (prompt_len // self.cache_stride + 1) * self.cache_stride
        kv_caches = self.init_cache(max_seq_len=initial_buffer_len)
        cache_len = initial_buffer_len
        params_signature = self._pytree_signature(self.params)
        platform_signature = self._platform_signature((self.params, prompt_tokens))
        while generated_count < max_new_tokens:
            # Every step, the kv-cache max length (`cache_len`) is set up to be a multiple of self.cache_stride.
            # `new_token_count` is bounded by both the remaining request and the available cache capacity.
            cache_capacity = cache_len - (prompt_len + generated_count)
            new_token_count = min(max_new_tokens - generated_count, cache_capacity)

            if new_token_count <= 0:
                cache_len += self.cache_stride
                kv_caches = tuple(c.resize(cache_len) for c in kv_caches)
                continue

            call_hash = _hash(
                GENERATION_AOT_CACHE_VERSION,
                str(self.model),
                str(self.config),
                params_signature,
                platform_signature,
                str(next_input_tokens.shape),
                str(sampling_method),
                str(top_k),
                str(cache_len),
                str(new_token_count),
            )

            step_output: GenerationOutput = generate(
                self.params,
                apply,
                next_input_tokens,
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
                skip_prefill=(generated_count > 0),
            )
            output_tokens.append(step_output.tokens)
            kv_caches = step_output.kv_caches
            generated_count += step_output.tokens.shape[1]
            next_input_tokens = step_output.tokens[:, -1:]

        if include_prompt:
            output_tokens.insert(0, prompt_tokens)
        return jnp.concatenate(output_tokens, axis=-1)
