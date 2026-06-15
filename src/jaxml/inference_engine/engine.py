import logging
import operator
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
from jaxml.inference_engine.sampling import SamplingMethod, normalize_sampling_params
from jaxml.outputs import GenerationOutput
from jaxml.utils import _hash, timeit

logger = logging.getLogger(__name__)

GENERATION_AOT_CACHE_VERSION = "generate_v4"


def _normalize_count(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e


def _normalize_bool(name: str, value: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value)}.")


def _normalize_optional_dtype(name: str, value: Any):
    if value is None:
        return None
    try:
        return jnp.dtype(value)
    except TypeError as e:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.") from e


def _normalize_dtype(name: str, value: Any):
    if value is None:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.")
    return _normalize_optional_dtype(name, value)


@struct.dataclass
class InferenceConfig:
    tp_size: int = 1
    dp_size: int = 1

    def __post_init__(self):
        tp_size = _normalize_count("tp_size", self.tp_size)
        dp_size = _normalize_count("dp_size", self.dp_size)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "dp_size", dp_size)
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}.")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}.")


def _normalize_inference_config(config: InferenceConfig) -> InferenceConfig:
    if not isinstance(config, InferenceConfig):
        raise TypeError(f"config must be an InferenceConfig, got {type(config)}.")
    return config


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
        config = _normalize_inference_config(config)
        dtype = _normalize_dtype("dtype", dtype)
        cache_stride = _normalize_count("cache_stride", cache_stride)
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
        max_seq_len = _normalize_count("max_seq_len", max_seq_len)
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        try:
            num_layers = self.model.config.num_layers
        except AttributeError as e:
            raise TypeError("model must expose config.num_layers to initialize KV caches.") from e
        num_layers = _normalize_count("model.config.num_layers", num_layers)
        if num_layers <= 0:
            raise ValueError(f"model.config.num_layers must be positive, got {num_layers}.")
        return tuple(KVCache.init(max_seq_len, None, None, dtype=self.dtype) for _ in range(num_layers))

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec, mesh: Optional[Mesh]) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    def _shard_params(self, x: Any, y: PartitionSpec):
        if not hasattr(y, "spec"):
            raise TypeError(f"Expected sharding object with a spec attribute, got {type(y)}.")
        if x.ndim != len(y.spec):
            if not (x.ndim == 2 and len(y.spec) == 3):
                raise ValueError(f"The shape of x ({x.shape}) and the sharding spec ({y.spec}) does not match.")
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

    @staticmethod
    def _validate_param_weights(weights: Any):
        if not isinstance(weights, dict):
            raise TypeError(f"weights must be a dict, got {type(weights)}.")
        if "params" not in weights:
            raise ValueError(f"The key 'params' was not found in weights. Got keys: {weights.keys()}.")

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
        use_tpu = _normalize_bool("use_tpu", use_tpu)
        reinit_weight = _normalize_bool("reinit_weight", reinit_weight)
        tp_size = self.config.tp_size
        dp_size = self.config.dp_size
        if weights is None:
            weights = self.params
        if not reinit_weight:
            self._validate_param_weights(weights)
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

        # Can assume weights is not None from now, and the goal is only to shard it.
        self._validate_param_weights(weights)
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
        dtype = _normalize_optional_dtype("dtype", dtype)

        def _prepare_leaf(x):
            try:
                x = jnp.asarray(x)
            except TypeError as e:
                raise TypeError(f"prepare_input leaves must be array-like, got {type(x)}.") from e
            if x.ndim != 2:
                raise ValueError(f"prepare_input leaves must be 2D arrays for data/model sharding, got shape {x.shape}.")
            if dtype is not None:
                x = x.astype(dtype)
            return x

        inputs = jax.tree.map(_prepare_leaf, inputs)

        mesh = Mesh(
            devices=mesh_utils.create_device_mesh((dp_size, tp_size)),
            axis_names=("data", "model"),
        )
        inputs = jax.device_put(inputs, self.mesh_sharding(PartitionSpec("data", None), mesh))
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
        if not jnp.issubdtype(prompt_tokens.dtype, jnp.integer):
            raise TypeError(f"prompt_tokens must contain integer token ids, got dtype {prompt_tokens.dtype}.")

        if prompt_tokens.shape[1] == 0:
            raise ValueError("prompt_tokens must contain at least one token.")

        if attention_mask is None:
            return prompt_tokens, None

        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None]
        elif attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be a 1D or 2D array, got shape {attention_mask.shape}.")
        if not (jnp.issubdtype(attention_mask.dtype, jnp.bool_) or jnp.issubdtype(attention_mask.dtype, jnp.integer)):
            raise TypeError(f"attention_mask must be boolean or integer, got dtype {attention_mask.dtype}.")

        if attention_mask.shape != prompt_tokens.shape:
            raise ValueError(
                f"attention_mask shape must match prompt_tokens shape; got {attention_mask.shape} and {prompt_tokens.shape}."
            )
        attention_mask = attention_mask.astype(bool)
        if not bool(jnp.all(jnp.any(attention_mask, axis=1))):
            raise ValueError("attention_mask must contain at least one valid token per batch row.")

        return prompt_tokens, attention_mask

    @staticmethod
    def _validate_generation_step_output(
        step_output: GenerationOutput,
        batch_size: int,
        max_new_tokens: int,
        expected_cache_count: int,
    ) -> tuple[jnp.ndarray, tuple[KVCache, ...], jnp.ndarray]:
        if not isinstance(step_output, GenerationOutput):
            raise TypeError(f"Internal generation must return a GenerationOutput, got {type(step_output)}.")

        tokens = jnp.asarray(step_output.tokens)
        if tokens.ndim != 2:
            raise ValueError(f"Internal generation tokens must be a 2D array, got shape {tokens.shape}.")
        if tokens.shape[0] != batch_size:
            raise ValueError(f"Internal generation token batch size must be {batch_size}, got {tokens.shape[0]}.")
        if tokens.shape[1] <= 0:
            raise ValueError("Internal generation must return at least one token per step.")
        if tokens.shape[1] > max_new_tokens:
            raise ValueError(f"Internal generation returned {tokens.shape[1]} tokens for a step limited to {max_new_tokens}.")
        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise TypeError(f"Internal generation tokens must contain integer token ids, got dtype {tokens.dtype}.")

        rng = step_output.rng
        if rng is None:
            raise ValueError("Internal generation did not return an RNG key for continuation.")
        rng = jnp.asarray(rng)
        if rng.shape != (2,):
            raise ValueError(f"Internal generation RNG must be a PRNG key with shape (2,), got shape {rng.shape}.")
        if not jnp.issubdtype(rng.dtype, jnp.integer):
            raise TypeError(f"Internal generation RNG must contain integer key data, got dtype {rng.dtype}.")

        try:
            kv_caches = tuple(step_output.kv_caches)
        except TypeError as e:
            raise TypeError("Internal generation kv_caches must be a sequence of KVCache instances.") from e
        if len(kv_caches) != expected_cache_count:
            raise ValueError(f"Internal generation returned {len(kv_caches)} KV caches, expected {expected_cache_count}.")
        for idx, kv_cache in enumerate(kv_caches):
            if not isinstance(kv_cache, KVCache):
                raise TypeError(
                    f"Internal generation kv_caches entries must be KVCache instances, got {type(kv_cache)} at index {idx}."
                )

        return tokens, kv_caches, rng

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
        seed = _normalize_count("seed", seed)
        max_new_tokens = _normalize_count("max_new_tokens", max_new_tokens)
        fuse_decoding = _normalize_bool("fuse_decoding", fuse_decoding)
        include_prompt = _normalize_bool("include_prompt", include_prompt)
        prompt_tokens, attention_mask = self._prepare_generation_inputs(prompt_tokens, attention_mask)
        apply = self.wrapped_apply_fn

        from .._generate import generate

        sampling_params = normalize_sampling_params(top_k=top_k, top_p=top_p, min_p=min_p, temp=temperature)
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p
        min_p = sampling_params.min_p
        temperature = sampling_params.temp
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
        rng = jax.random.PRNGKey(seed)
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
                attention_mask if generated_count == 0 else None,
                kv_caches,
                call_hash,
                sampling_method,
                seed=seed,
                rng=rng,
                max_new_tokens=new_token_count,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
                fuse_decoding=fuse_decoding,
                include_prompt=False,
                skip_prefill=(generated_count > 0),
            )
            step_tokens, kv_caches, rng = self._validate_generation_step_output(
                step_output,
                batch_size=prompt_tokens.shape[0],
                max_new_tokens=new_token_count,
                expected_cache_count=len(kv_caches),
            )
            output_tokens.append(step_tokens)
            generated_count += step_tokens.shape[1]
            next_input_tokens = step_tokens[:, -1:]

        if include_prompt:
            output_tokens.insert(0, prompt_tokens)
        return jnp.concatenate(output_tokens, axis=-1)
