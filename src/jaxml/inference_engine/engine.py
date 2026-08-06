import functools
import inspect
import logging
import operator
import warnings
from collections.abc import Mapping
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

from jaxml._validation import contains_tracer as _contains_tracer
from jaxml.cache import KVCache, is_model_cache, model_cache_batch_size, model_cache_is_populated
from jaxml.inference_engine.sampling import SamplingMethod, normalize_sampling_params
from jaxml.outputs import GenerationOutput
from jaxml.utils import _hash, timeit

logger = logging.getLogger(__name__)

GENERATION_AOT_CACHE_VERSION = "generate_v7"


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
        return jax.dtypes.canonicalize_dtype(jnp.dtype(value))
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.") from e


def _normalize_dtype(name: str, value: Any):
    if value is None:
        raise TypeError(f"{name} must be a valid JAX dtype, got {value!r}.")
    return _normalize_optional_dtype(name, value)


@struct.dataclass
class InferenceConfig:
    """Sharding spec by axis sizes. The mesh is ``(data=dp_size, model=tp_size,
    expert=ep_size)``; a size > 1 turns on that parallelism (data/tensor/expert)
    via the shared :data:`SHARDING_RULES`. The product must equal the device count.
    """

    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1

    def __post_init__(self):
        tp_size = _normalize_count("tp_size", self.tp_size)
        dp_size = _normalize_count("dp_size", self.dp_size)
        ep_size = _normalize_count("ep_size", self.ep_size)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "dp_size", dp_size)
        object.__setattr__(self, "ep_size", ep_size)
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}.")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}.")
        if ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {ep_size}.")

    @property
    def num_devices(self) -> int:
        return self.tp_size * self.dp_size * self.ep_size

    @property
    def mesh_layout(self) -> tuple[int, int, int]:
        return (self.dp_size, self.tp_size, self.ep_size)


def _normalize_inference_config(config: InferenceConfig) -> InferenceConfig:
    if not isinstance(config, InferenceConfig):
        raise TypeError(f"config must be an InferenceConfig, got {type(config)}.")
    return config


# Logical mesh axes, ordered to match InferenceConfig.mesh_layout = (dp, tp, ep).
MESH_AXIS_NAMES = ("data", "model", "expert")

# Single source of truth mapping every logical tensor axis to a mesh axis (or None =
# replicated). A non-trivial axis size in InferenceConfig turns the corresponding
# parallelism on: batch->data is DDP, the model-axis group is tensor/model parallelism,
# expert->expert is expert parallelism. embed->None / length->None leave room for FSDP
# (embed->data) and sequence parallelism (length->...) without changing the notation.
SHARDING_RULES = (
    ("batch", "data"),
    ("heads", "model"),
    ("heads_merged", "model"),
    ("intermediate", "model"),
    ("vocab", "model"),
    ("expert", "expert"),
    ("embed", None),
    ("length", None),
    ("kv_length", None),
    ("head_states", None),
    ("qkv", None),
    ("q_lora", None),
    ("kv_lora", None),
    ("kv_lora_rope", None),
)


def _logical_axes_in_spec(partition_spec_tree: Any) -> set:
    """Collect every logical axis name appearing in a get_partition_spec pytree."""
    axes = set()
    for leaf in jax.tree.leaves(partition_spec_tree, is_leaf=lambda x: isinstance(x, PartitionSpec)):
        if isinstance(leaf, PartitionSpec):
            for entry in leaf:
                if entry is None:
                    continue
                for name in (entry if isinstance(entry, tuple) else (entry,)):
                    if name is not None:
                        axes.add(name)
    return axes


def _validate_rules_cover_axes(partition_spec_tree: Any, rules=SHARDING_RULES):
    """Fail loudly if a model annotates a logical axis with no sharding rule.

    Without a rule, logical_to_mesh_sharding silently replicates the axis, which is a
    hard-to-spot correctness/performance footgun (e.g. an unsharded expert axis).
    """
    known = {name for name, _ in rules}
    missing = _logical_axes_in_spec(partition_spec_tree) - known
    if missing:
        raise ValueError(
            f"Model uses logical axes with no entry in SHARDING_RULES: {sorted(missing)}. "
            "Add them to jaxml.inference_engine.engine.SHARDING_RULES."
        )


class Engine:
    """Wrap around a model class to do autoregressive generation."""

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        params: FrozenDict,
        dtype: Any = None,
        cache_stride: int = 256,
    ):
        config = _normalize_inference_config(config)
        if dtype is None:
            dtype = getattr(model, "dtype", None) or jnp.float32
        dtype = _normalize_dtype("dtype", dtype)
        cache_stride = _normalize_count("cache_stride", cache_stride)
        if cache_stride <= 0:
            raise ValueError(f"cache_stride must be positive, got {cache_stride}.")
        required_devices = config.num_devices
        available_devices = jax.device_count()
        if required_devices != available_devices:
            raise ValueError(
                f"InferenceConfig sharding requires exactly {required_devices} devices "
                f"(dp_size={config.dp_size}, tp_size={config.tp_size}, ep_size={config.ep_size}), "
                f"but {available_devices} are available."
            )
        self.model = model
        self.config = config
        self.params = params
        self.dtype = dtype
        self.cache_stride = cache_stride
        try:
            model_call_parameters = inspect.signature(model.__call__).parameters
        except (AttributeError, TypeError, ValueError):
            model_call_parameters = {}
        self._supports_logits_positions = "logits_positions" in model_call_parameters

    def init_cache(self, max_seq_len: int) -> tuple:
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
        model_cache_init = getattr(self.model, "init_cache", None)
        if callable(model_cache_init):
            caches = tuple(model_cache_init(max_seq_len=max_seq_len, dtype=self.dtype))
            if len(caches) != num_layers:
                raise ValueError(
                    f"model.init_cache must return one cache per layer, got {len(caches)} and expected {num_layers}."
                )
            for idx, cache in enumerate(caches):
                if not is_model_cache(cache):
                    raise TypeError(
                        f"model.init_cache entries must implement the model cache protocol, got {type(cache)} at index {idx}."
                    )
                cache.validate_state()
            return caches
        return tuple(KVCache.init(max_seq_len, None, None, dtype=self.dtype) for _ in range(num_layers))

    def _model_vocab_size(self) -> int:
        try:
            vocab_size = self.model.config.vocab_size
        except AttributeError as e:
            raise TypeError("model must expose config.vocab_size to validate generation token ids.") from e
        vocab_size = _normalize_count("model.config.vocab_size", vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"model.config.vocab_size must be positive, got {vocab_size}.")
        return vocab_size

    def _model_max_position_embeddings(self) -> Optional[int]:
        """Return the model context limit when the adapter exposes one."""
        config = getattr(self.model, "config", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", None)
        if max_position_embeddings is None:
            # Preserve support for custom adapters that predate this optional
            # part of the Engine model contract.
            return None
        max_position_embeddings = _normalize_count(
            "model.config.max_position_embeddings",
            max_position_embeddings,
        )
        if max_position_embeddings <= 0:
            raise ValueError(f"model.config.max_position_embeddings must be positive, got {max_position_embeddings}.")
        return max_position_embeddings

    @staticmethod
    def _validate_token_ids_in_vocab(name: str, token_ids: jnp.ndarray, vocab_size: int) -> jnp.ndarray:
        if not _contains_tracer(token_ids) and bool(jnp.any((token_ids < 0) | (token_ids >= vocab_size))):
            raise ValueError(f"{name} token ids must be within [0, {vocab_size}).")
        return token_ids

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec, mesh: Optional[Mesh]) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    def _create_configured_mesh(self) -> Mesh:
        mesh_layout = self.config.mesh_layout  # (dp, tp, ep)
        required_devices = self.config.num_devices
        devices = tuple(jax.devices()[:required_devices])
        if len(devices) != required_devices:
            raise ValueError(
                f"InferenceConfig requires {required_devices} devices "
                f"(dp_size={self.config.dp_size}, tp_size={self.config.tp_size}, "
                f"ep_size={self.config.ep_size}), but only {len(devices)} are available."
            )
        return Mesh(
            devices=mesh_utils.create_device_mesh(mesh_layout, devices=devices),
            axis_names=MESH_AXIS_NAMES,
        )

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
    def _device_platform(device: Any) -> Optional[str]:
        if callable(device):
            device = device()
        if device is None:
            return None
        return str(getattr(device, "platform", device))

    @staticmethod
    def _platform_signature(tree: Any) -> str:
        import jaxlib

        platforms = set()
        for leaf in jax.tree.leaves(tree):
            platform = Engine._device_platform(getattr(leaf, "device", None))
            if platform is not None:
                platforms.add(platform)
            for shard in getattr(leaf, "addressable_shards", ()):
                platform = Engine._device_platform(getattr(shard, "device", None))
                if platform is not None:
                    platforms.add(platform)
        if not platforms:
            platforms.add(jax.default_backend())
        return str((jax.__version__, jaxlib.__version__, jax.default_backend(), tuple(sorted(platforms))))

    @staticmethod
    def _validate_param_weights(weights: Any):
        if not isinstance(weights, Mapping):
            raise TypeError(f"weights must be a mapping, got {type(weights)}.")
        if "params" not in weights:
            raise ValueError(f"The key 'params' was not found in weights. Got keys: {weights.keys()}.")
        if not isinstance(weights["params"], Mapping):
            raise TypeError(f"weights['params'] must be a mapping, got {type(weights['params'])}.")

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
        if weights is None:
            weights = self.params
        if not reinit_weight:
            self._validate_param_weights(weights)
        # whether: use 1 single TPU or CPU
        is_single_device = (self.config.num_devices == 1) or not use_tpu

        key = jax.random.PRNGKey(0)

        # Tiny (dp, tp)-shaped token batch only used to trace the model for shapes.
        dummy_input = jnp.array([[1 for _ in range(self.config.tp_size)] for _ in range(self.config.dp_size)])
        abstract_variables = jax.eval_shape(self.model.init, key, dummy_input)

        if not is_single_device:
            mesh = self._create_configured_mesh()

            logical_state_spec = nn.get_partition_spec(abstract_variables)
            _validate_rules_cover_axes(logical_state_spec)
            logical_state_sharding = logical_to_mesh_sharding(logical_state_spec, mesh, SHARDING_RULES)
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
        dtype = _normalize_optional_dtype("dtype", dtype)

        def _prepare_leaf(x):
            try:
                x = jnp.asarray(x)
            except TypeError as e:
                raise TypeError(f"prepare_input leaves must be array-like, got {type(x)}.") from e
            if x.ndim != 2:
                raise ValueError(f"prepare_input leaves must be 2D arrays for data/model sharding, got shape {x.shape}.")
            if x.shape[0] == 0:
                raise ValueError("prepare_input leaves must contain at least one batch row.")
            if x.shape[1] == 0:
                raise ValueError("prepare_input leaves must contain at least one token.")
            if x.shape[0] % self.config.dp_size != 0:
                raise ValueError(
                    f"prepare_input batch size must be divisible by dp_size={self.config.dp_size}, got {x.shape[0]}."
                )
            if dtype is not None:
                x = x.astype(dtype)
            return x

        inputs = jax.tree.map(_prepare_leaf, inputs)
        leaves = jax.tree.leaves(inputs)
        if not leaves:
            raise ValueError("prepare_input inputs must contain at least one array leaf.")
        batch_size = leaves[0].shape[0]
        for leaf in leaves[1:]:
            if leaf.shape[0] != batch_size:
                raise ValueError(f"prepare_input leaves must share the same batch size, got {batch_size} and {leaf.shape[0]}.")

        mesh = self._create_configured_mesh()
        inputs = jax.device_put(inputs, self.mesh_sharding(PartitionSpec("data", None), mesh))
        return inputs

    def wrapped_apply_fn(
        self,
        params,
        input_token,
        attention_mask=None,
        kv_caches=None,
        use_cache=True,
        gather_prefill_logits=False,
        prefill_logits_tail=1,
    ) -> tuple[jnp.ndarray, tuple]:

        gather_prefill_logits = _normalize_bool("gather_prefill_logits", gather_prefill_logits)
        gather_last_valid_logit = gather_prefill_logits and attention_mask is not None
        prefill_logits_tail = _normalize_count("prefill_logits_tail", prefill_logits_tail)
        if prefill_logits_tail <= 0:
            raise ValueError(f"prefill_logits_tail must be positive, got {prefill_logits_tail}.")
        effective_logits_tail = prefill_logits_tail if gather_last_valid_logit else 1
        if effective_logits_tail > input_token.shape[1]:
            raise ValueError(
                "prefill_logits_tail must not exceed the prefill token length; "
                f"got {effective_logits_tail} and {input_token.shape[1]}."
            )
        logits_positions = None
        if gather_last_valid_logit:
            positions = jnp.arange(attention_mask.shape[1])[None, :]
            logits_positions = jnp.max(jnp.where(attention_mask, positions, -1), axis=1)

        use_logits_positions = gather_last_valid_logit and self._supports_logits_positions
        keep_last_n_logits = 0 if use_logits_positions else effective_logits_tail
        model_kwargs = {}
        if self._supports_logits_positions:
            model_kwargs["logits_positions"] = logits_positions if use_logits_positions else None

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
            keep_last_n_logits=keep_last_n_logits,
            **model_kwargs,
        )  # return a tuple (CausalLMOutputWithCache, dict) where dict is the mutable cache

        logits = out.logits
        if gather_last_valid_logit and not self._supports_logits_positions:
            tail_start = input_token.shape[1] - effective_logits_tail
            tail_positions = logits_positions - tail_start
            logits = logits[jnp.arange(logits.shape[0]), tail_positions][:, None, :]

        return logits, out.kv_caches

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

        if prompt_tokens.shape[0] == 0:
            raise ValueError("prompt_tokens must contain at least one batch row.")
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
        if jnp.issubdtype(attention_mask.dtype, jnp.integer) and not _contains_tracer(attention_mask):
            if bool(jnp.any((attention_mask != 0) & (attention_mask != 1))):
                raise ValueError("attention_mask integer values must be 0 or 1.")
        attention_mask = attention_mask.astype(bool)
        if not _contains_tracer(attention_mask) and not bool(jnp.all(jnp.any(attention_mask, axis=1))):
            raise ValueError("attention_mask must contain at least one valid token per batch row.")
        if not _contains_tracer(attention_mask) and attention_mask.shape[1] > 1:
            has_valid_token_after_padding = jnp.any(~attention_mask[:, :-1] & attention_mask[:, 1:])
            if bool(has_valid_token_after_padding):
                raise ValueError("attention_mask must be right padded: valid tokens cannot appear after masked positions.")

        return prompt_tokens, attention_mask

    @staticmethod
    def _validate_generation_step_output(
        step_output: GenerationOutput,
        batch_size: int,
        max_new_tokens: int,
        expected_cache_count: int,
        vocab_size: int,
    ) -> tuple[jnp.ndarray, tuple, jnp.ndarray]:
        if not isinstance(step_output, GenerationOutput):
            raise TypeError(f"Internal generation must return a GenerationOutput, got {type(step_output)}.")

        tokens = jnp.asarray(step_output.tokens)
        if tokens.ndim != 2:
            raise ValueError(f"Internal generation tokens must be a 2D array, got shape {tokens.shape}.")
        if tokens.shape[0] != batch_size:
            raise ValueError(f"Internal generation token batch size must be {batch_size}, got {tokens.shape[0]}.")
        if tokens.shape[1] <= 0:
            raise ValueError("Internal generation must return at least one token per step.")
        if tokens.shape[1] != max_new_tokens:
            raise ValueError(
                "Internal generation token length must match requested step length; "
                f"got {tokens.shape[1]} and expected {max_new_tokens}."
            )
        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise TypeError(f"Internal generation tokens must contain integer token ids, got dtype {tokens.dtype}.")
        tokens = Engine._validate_token_ids_in_vocab("Internal generation", tokens, vocab_size)

        rng = step_output.rng
        if rng is None:
            raise ValueError("Internal generation did not return an RNG key for continuation.")
        rng = jnp.asarray(rng)
        if rng.shape != (2,):
            raise ValueError(f"Internal generation RNG must be a PRNG key with shape (2,), got shape {rng.shape}.")
        if rng.dtype != jnp.uint32:
            raise TypeError(f"Internal generation RNG must contain uint32 key data, got dtype {rng.dtype}.")

        try:
            kv_caches = tuple(step_output.kv_caches)
        except TypeError as e:
            raise TypeError("Internal generation kv_caches must be a sequence of KVCache instances.") from e
        if len(kv_caches) != expected_cache_count:
            raise ValueError(f"Internal generation returned {len(kv_caches)} KV caches, expected {expected_cache_count}.")
        for idx, kv_cache in enumerate(kv_caches):
            if not is_model_cache(kv_cache):
                raise TypeError(
                    f"Internal generation kv_caches entries must be KVCache instances, got {type(kv_cache)} at index {idx}."
                )
            try:
                kv_cache.validate_state()
            except (TypeError, ValueError) as e:
                raise type(e)(f"Invalid internal generation KV cache at index {idx}: {e}") from e
            cache_batch_size = model_cache_batch_size(kv_cache)
            if cache_batch_size is not None and cache_batch_size != batch_size:
                raise ValueError(
                    "Internal generation KV cache batch size must match generated token batch size; "
                    f"got {cache_batch_size} and {batch_size} at index {idx}."
                )

        return tokens, kv_caches, rng

    @staticmethod
    def _validate_prefilled_generation_caches(kv_caches: tuple):
        for idx, kv_cache in enumerate(kv_caches):
            if not model_cache_is_populated(kv_cache):
                raise ValueError(
                    f"Internal generation must return prefilled KV caches before continuation; got empty cache at index {idx}."
                )

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
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
        fuse_decoding = _normalize_bool("fuse_decoding", fuse_decoding)
        include_prompt = _normalize_bool("include_prompt", include_prompt)
        prompt_tokens, attention_mask = self._prepare_generation_inputs(prompt_tokens, attention_mask)
        vocab_size = self._model_vocab_size()
        prompt_tokens = self._validate_token_ids_in_vocab("prompt_tokens", prompt_tokens, vocab_size)

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
        max_valid_prompt_len = prompt_len
        if attention_mask is not None and not _contains_tracer(attention_mask):
            max_valid_prompt_len = int(jnp.max(jnp.sum(attention_mask, axis=1)))
        context_limit = self._model_max_position_embeddings()
        if context_limit is not None and prompt_len > context_limit:
            raise ValueError(
                f"physical prompt length must not exceed model.config.max_position_embeddings={context_limit}; "
                f"got {prompt_len}."
            )
        requested_sequence_len = max_valid_prompt_len + max_new_tokens
        if context_limit is not None and requested_sequence_len > context_limit:
            raise ValueError(
                "maximum valid prompt length plus max_new_tokens must not exceed "
                f"model.config.max_position_embeddings={context_limit}; got "
                f"{max_valid_prompt_len} + {max_new_tokens} = {requested_sequence_len}."
            )

        if max_new_tokens == 0:
            if include_prompt:
                return prompt_tokens
            return jnp.empty((prompt_tokens.shape[0], 0), dtype=prompt_tokens.dtype)

        # The model normally computes only the physical last prompt logit. For a
        # right-padded batch, that position belongs to padding in shorter rows, so
        # gather each row's last valid hidden state before the LM head. Keep the
        # one-logit fast path for unmasked and fully valid batches.
        gather_prefill_logits = attention_mask is not None and (
            _contains_tracer(attention_mask) or not bool(jnp.all(attention_mask[:, -1]))
        )
        prefill_logits_tail = 1
        if gather_prefill_logits and not self._supports_logits_positions:
            if _contains_tracer(attention_mask):
                prefill_logits_tail = prompt_len
            else:
                positions = jnp.arange(prompt_len)[None, :]
                last_valid_positions = jnp.max(jnp.where(attention_mask, positions, -1), axis=1)
                prefill_logits_tail = prompt_len - int(jnp.min(last_valid_positions))

        prepared_inputs = {"prompt_tokens": prompt_tokens}
        if attention_mask is not None:
            prepared_inputs["attention_mask"] = attention_mask
        prepared_inputs = self.prepare_input(prepared_inputs)
        prompt_tokens = prepared_inputs["prompt_tokens"]
        attention_mask = prepared_inputs.get("attention_mask")

        output_tokens = []
        next_input_tokens = prompt_tokens
        generated_count = 0
        rng = jax.random.PRNGKey(seed)
        initial_buffer_len = (prompt_len // self.cache_stride + 1) * self.cache_stride
        if context_limit is not None:
            # Stride rounding can otherwise exceed a non-aligned cached RoPE
            # table even when the actual token request remains in context.
            initial_buffer_len = min(initial_buffer_len, context_limit)
        kv_caches = self.init_cache(max_seq_len=initial_buffer_len)
        cache_len = initial_buffer_len
        params_signature = self._pytree_signature(self.params)
        platform_signature = self._platform_signature((self.params, prompt_tokens))
        while generated_count < max_new_tokens:
            # Every step, the kv-cache max length (`cache_len`) is set up to be a multiple of self.cache_stride.
            # `new_token_count` is bounded by both the remaining request and the available cache capacity.
            cache_capacity = cache_len - (max_valid_prompt_len + generated_count)
            new_token_count = min(max_new_tokens - generated_count, cache_capacity)

            if new_token_count <= 0:
                next_cache_len = cache_len + self.cache_stride
                if context_limit is not None:
                    next_cache_len = min(next_cache_len, context_limit)
                if next_cache_len <= cache_len:
                    raise RuntimeError("KV cache cannot grow enough to satisfy the validated generation request.")
                cache_len = next_cache_len
                kv_caches = tuple(c.resize(cache_len) for c in kv_caches)
                continue

            step_attention_mask = attention_mask if generated_count == 0 else None
            step_gather_prefill_logits = gather_prefill_logits and step_attention_mask is not None
            step_prefill_logits_tail = prefill_logits_tail if step_gather_prefill_logits else 1
            apply = functools.partial(
                self.wrapped_apply_fn,
                gather_prefill_logits=step_gather_prefill_logits,
                prefill_logits_tail=step_prefill_logits_tail,
            )
            input_signature = self._pytree_signature((next_input_tokens, step_attention_mask))
            kv_cache_signature = self._pytree_signature(kv_caches)
            call_hash = _hash(
                GENERATION_AOT_CACHE_VERSION,
                str(self.model),
                str(self.config),
                params_signature,
                platform_signature,
                input_signature,
                kv_cache_signature,
                str(self._supports_logits_positions),
                str(step_gather_prefill_logits),
                str(step_prefill_logits_tail),
                str(sampling_method),
                str(top_k),
                str(top_p),
                str(min_p),
                str(temperature),
                str(cache_len),
                # A fused scan specializes on its length. The non-fused path
                # compiles one decode step, so chunk length must not multiply
                # otherwise identical AOT artifacts.
                str(new_token_count) if fuse_decoding else "single_step",
            )

            step_output: GenerationOutput = generate(
                self.params,
                apply,
                next_input_tokens,
                step_attention_mask,
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
                vocab_size=vocab_size,
            )
            output_tokens.append(step_tokens)
            generated_count += step_tokens.shape[1]
            if generated_count < max_new_tokens:
                self._validate_prefilled_generation_caches(kv_caches)
            next_input_tokens = step_tokens[:, -1:]

        if include_prompt:
            output_tokens.insert(0, prompt_tokens)
        return jnp.concatenate(output_tokens, axis=-1)
