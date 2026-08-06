import json
import re
from collections.abc import Mapping
from operator import index
from os import PathLike, fspath
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from jaxml.models.deepseek import DeepseekModelWithHead
from jaxml.models.gemma3 import GemmaModelWithHead
from jaxml.models.gpt_neox import GPTNeoXModelWithHead
from jaxml.models.kimi_linear import KimiLinearModelWithHead
from jaxml.models.llama import LlamaModelWithHead

from .utils import _resolve_np_dtype, _state_value_to_numpy, torch_to_jax_states


class _LazySafetensorsStateDict(Mapping):
    """Read one checkpoint shard at a time through the state-dict interface."""

    def __init__(self, weight_map: Mapping[str, str], resolve_shard):
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("Safetensors checkpoint index must contain a non-empty weight_map.")
        self._weight_map = dict(weight_map)
        self._resolve_shard = resolve_shard
        self._loaded_name = None
        self._loaded_state = {}

    def __len__(self):
        return len(self._weight_map)

    def __iter__(self):
        return iter(self._weight_map)

    def __contains__(self, key):
        return key in self._weight_map

    def __getitem__(self, key):
        try:
            shard_name = self._weight_map[key]
        except KeyError as e:
            raise KeyError(key) from e
        if shard_name != self._loaded_name:
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError("Please install safetensors to load Kimi Linear checkpoints.") from e
            shard_path = self._resolve_shard(shard_name)
            self._loaded_state = load_file(shard_path, device="cpu")
            self._loaded_name = shard_name
        try:
            return self._loaded_state[key]
        except KeyError as e:
            raise KeyError(f"Safetensors shard {shard_name!r} does not contain indexed tensor {key!r}.") from e


def _load_kimi_linear_safetensors_source(name: str, **from_pretrained_kwargs):
    """Build a config/state-dict facade without importing Kimi's Torch/FLA model."""
    kwargs = dict(from_pretrained_kwargs)
    subfolder = kwargs.pop("subfolder", None)
    if subfolder is not None and not isinstance(subfolder, str):
        raise TypeError(f"subfolder must be a string when set, got {type(subfolder)}.")

    hub_keys = {
        "revision",
        "cache_dir",
        "local_dir",
        "user_agent",
        "force_download",
        "token",
        "local_files_only",
        "headers",
        "endpoint",
    }
    ignored_model_keys = {
        "attn_implementation",
        "device_map",
        "low_cpu_mem_usage",
        "torch_dtype",
        "trust_remote_code",
        "weights_only",
    }
    use_safetensors = kwargs.pop("use_safetensors", True)
    if use_safetensors is False:
        raise ValueError("Kimi Linear direct loading requires use_safetensors=True.")
    unexpected = set(kwargs) - hub_keys - ignored_model_keys
    if unexpected:
        raise TypeError(
            "Unsupported Kimi Linear checkpoint loading keyword arguments: "
            f"{sorted(unexpected)}. Pass hub download options or model dtype through the loader's dtype argument."
        )
    hub_kwargs = {key: kwargs[key] for key in hub_keys if key in kwargs}

    local_root = Path(name)
    if local_root.is_dir():
        if subfolder:
            local_root = local_root / subfolder

        def resolve_file(filename):
            path = local_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Kimi Linear checkpoint file does not exist: {path}.")
            return str(path)

    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("Please install huggingface-hub to download Kimi Linear checkpoints.") from e

        def resolve_file(filename):
            return str(
                hf_hub_download(
                    repo_id=name,
                    filename=filename,
                    subfolder=subfolder,
                    **hub_kwargs,
                )
            )

    config_path = Path(resolve_file("config.json"))
    index_path = Path(resolve_file("model.safetensors.index.json"))
    try:
        config_data = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read Kimi Linear config from {config_path}.") from e
    try:
        index_data = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read Kimi Linear checkpoint index from {index_path}.") from e
    if not isinstance(config_data, dict):
        raise ValueError(f"Kimi Linear config must be a JSON object, got {type(config_data)}.")
    weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
    state_dict = _LazySafetensorsStateDict(weight_map, resolve_file)
    config = SimpleNamespace(**config_data)
    return SimpleNamespace(config=config, state_dict=lambda: state_dict)


def _normalize_hf_count(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value)}.")
    try:
        normalized = index(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(value)}.") from e
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _config_head_dim(config):
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return _normalize_hf_count("head_dim", head_dim)

    hidden_size = _normalize_hf_count("hidden_size", config.hidden_size)
    num_attention_heads = _normalize_hf_count("num_attention_heads", config.num_attention_heads)
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is not set.")
    return hidden_size // num_attention_heads


def _hf_model_config(model, architecture_name: str):
    try:
        return model.config
    except AttributeError as e:
        raise ValueError(f"Expected Hugging Face {architecture_name} model to expose a config.") from e


def _hf_model_state_dict(model, architecture_name: str) -> Mapping:
    state_dict_fn = getattr(model, "state_dict", None)
    if not callable(state_dict_fn):
        raise TypeError(f"Expected Hugging Face {architecture_name} model to expose a callable state_dict method.")
    state_dict = state_dict_fn()
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Hugging Face {architecture_name} state_dict must be a mapping, got {type(state_dict)}.")
    return state_dict


def to_llama_jax_params(model, dtype: str = "float16"):
    llama_config = _hf_model_config(model, "Llama")
    return torch_to_jax_states(_hf_model_state_dict(model, "Llama"), head_dim=_config_head_dim(llama_config), dtype=dtype)


def to_mistral_jax_params(model, dtype: str = "float16"):
    # Mistral shares Llama's parameter naming, so the Llama conversion applies unchanged.
    mistral_config = _hf_model_config(model, "Mistral")
    return torch_to_jax_states(_hf_model_state_dict(model, "Mistral"), head_dim=_config_head_dim(mistral_config), dtype=dtype)


def to_gemma_jax_params(model, dtype: str = "bfloat16"):
    gemma_config = _hf_model_config(model, "Gemma")
    return torch_to_jax_states(_hf_model_state_dict(model, "Gemma"), head_dim=_config_head_dim(gemma_config), dtype=dtype)


def to_neox_jax_params(model, dtype: str = "float16"):
    neox_config = _hf_model_config(model, "GPT-NeoX")
    # the substitution rules may be fragile, but we specifically target HF's neox
    _sub = {
        r"embed_in(.*)": r"embed_tokens\1",
        r"final_layer_norm(.*)": r"norm\1",
        r"embed_out(.*)": r"lm_head\1",
        r"attention\.(.*)": r"self_attn.\1",
        r"query_key_value(.*)": r"qkv_proj\1",
        r"dense\.(.*)": r"o_proj.\1",
        r"dense_h_to_4h(.*)": r"up_proj\1",
        r"dense_4h_to_h(.*)": r"down_proj\1",
    }
    state_dict = {}
    state_sources = {}
    for key, value in _hf_model_state_dict(model, "GPT-NeoX").items():
        name = key
        for k, v in _sub.items():
            name = re.sub(k, v, name)
        if name in state_dict:
            raise ValueError(
                "Multiple Hugging Face GPT-NeoX state keys map to the same jaxml destination "
                f"{name!r}: {state_sources[name]!r} and {key!r}."
            )
        state_dict[name] = value
        state_sources[name] = key
    return torch_to_jax_states(state_dict, head_dim=_config_head_dim(neox_config), dtype=dtype)


def to_deepseek_jax_params(model, dtype: str = "bfloat16"):
    """Convert an HF MLA/MoE causal LM into the jaxml DeepseekModelWithHead tree.

    DeepSeek-V3, GLM-4.7-Flash, and Kimi-K2 share the MLA projection and
    fine-grained MoE parameter layout used by :class:`DeepseekModelWithHead`.
    Hugging Face may expose routed experts as fused 3D tensors, while raw
    checkpoints may contain one gate/up/down tensor per expert; both forms are
    accepted here.
    """
    from jaxml.config import ModelConfig

    hf_config = _hf_model_config(model, "MLA/MoE")
    architecture_name = {
        "deepseek_v3": "DeepSeek",
        "glm4_moe_lite": "GLM",
        "kimi_k2": "Kimi",
    }.get(getattr(hf_config, "model_type", None), "MLA/MoE")
    cfg = ModelConfig.from_hf(hf_config)
    state_dict = _hf_model_state_dict(model, architecture_name)
    np_dtype = _resolve_np_dtype(dtype)

    def g(key):
        try:
            value = state_dict[key]
        except KeyError as e:
            raise KeyError(f"Hugging Face {architecture_name} state dict is missing required tensor {key!r}.") from e
        return _state_value_to_numpy(value, np_dtype, key)

    nh, qk = cfg.num_heads, cfg.qk_head_dim
    nope, v = cfg.qk_nope_head_dim, cfg.v_head_dim

    def attn_params(prefix):
        attn = {}
        if cfg.q_lora_rank is None:
            attn["q_proj"] = {"kernel": g(prefix + "q_proj.weight").T.reshape(-1, nh, qk)}
        else:
            attn["q_a_proj"] = {"kernel": g(prefix + "q_a_proj.weight").T}
            attn["q_a_layernorm"] = {"weight": g(prefix + "q_a_layernorm.weight")}
            attn["q_b_proj"] = {"kernel": g(prefix + "q_b_proj.weight").T.reshape(-1, nh, qk)}
        attn["kv_a_proj_with_mqa"] = {"kernel": g(prefix + "kv_a_proj_with_mqa.weight").T}
        attn["kv_a_layernorm"] = {"weight": g(prefix + "kv_a_layernorm.weight")}
        attn["kv_b_proj"] = {"kernel": g(prefix + "kv_b_proj.weight").T.reshape(-1, nh, nope + v)}
        attn["o_proj"] = {"kernel": g(prefix + "o_proj.weight").T}
        return attn

    def expert_params(prefix):
        fused_gate_up_key = prefix + "experts.gate_up_proj"
        fused_down_key = prefix + "experts.down_proj"
        has_fused_gate_up = fused_gate_up_key in state_dict
        has_fused_down = fused_down_key in state_dict
        if has_fused_gate_up or has_fused_down:
            if not (has_fused_gate_up and has_fused_down):
                raise ValueError(
                    f"Hugging Face {architecture_name} state dict must contain both fused expert tensors "
                    f"{fused_gate_up_key!r} and {fused_down_key!r}."
                )
            gate_up = g(fused_gate_up_key)
            down = g(fused_down_key)
        else:
            gate_up_experts = []
            down_experts = []
            for expert_idx in range(cfg.n_routed_experts):
                expert_prefix = prefix + f"experts.{expert_idx}."
                gate = g(expert_prefix + "gate_proj.weight")
                up = g(expert_prefix + "up_proj.weight")
                if gate.shape != up.shape:
                    raise ValueError(
                        f"Hugging Face {architecture_name} expert {expert_idx} gate/up shapes must match, "
                        f"got {gate.shape} and {up.shape}."
                    )
                gate_up_experts.append(np.concatenate((gate, up), axis=0))
                down_experts.append(g(expert_prefix + "down_proj.weight"))
            gate_up = np.stack(gate_up_experts, axis=0)
            down = np.stack(down_experts, axis=0)
        return gate_up.transpose(0, 2, 1), down.transpose(0, 2, 1)

    def mlp_params(prefix, is_moe):
        if not is_moe:
            return {
                "gate_proj": {"kernel": g(prefix + "gate_proj.weight").T},
                "up_proj": {"kernel": g(prefix + "up_proj.weight").T},
                "down_proj": {"kernel": g(prefix + "down_proj.weight").T},
            }
        gate_up_kernel, down_kernel = expert_params(prefix)
        return {
            "gate_kernel": g(prefix + "gate.weight").T,
            "e_score_correction_bias": g(prefix + "gate.e_score_correction_bias"),
            "gate_up_kernel": gate_up_kernel,
            "down_kernel": down_kernel,
            "shared_gate_proj": {"kernel": g(prefix + "shared_experts.gate_proj.weight").T},
            "shared_up_proj": {"kernel": g(prefix + "shared_experts.up_proj.weight").T},
            "shared_down_proj": {"kernel": g(prefix + "shared_experts.down_proj.weight").T},
        }

    def is_sparse_moe_layer(layer_idx):
        return cfg.is_moe and layer_idx >= cfg.first_k_dense_replace and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0

    model_params = {"embed_tokens": {"embedding": g("model.embed_tokens.weight")}}
    for idx in range(cfg.num_layers):
        layer_prefix = f"model.layers.{idx}."
        model_params[f"layers_{idx}"] = {
            "self_attn": attn_params(layer_prefix + "self_attn."),
            "mlp": mlp_params(layer_prefix + "mlp.", is_sparse_moe_layer(idx)),
            "input_layernorm": {"weight": g(layer_prefix + "input_layernorm.weight")},
            "post_attention_layernorm": {"weight": g(layer_prefix + "post_attention_layernorm.weight")},
        }
    model_params["norm"] = {"weight": g("model.norm.weight")}

    return {"params": {"model": model_params, "lm_head": {"kernel": g("lm_head.weight").T}}}


def to_kimi_linear_jax_params(model, dtype: str = "bfloat16"):
    """Convert an official Kimi Linear causal LM checkpoint to jaxml.

    Kimi Linear combines KDA and NoPE MLA attention layers. Its routed expert
    checkpoint uses the original per-expert ``w1``/``w2``/``w3`` layout, so
    those tensors are fused into the layout shared by :class:`DeepseekMoE`.
    """
    from jaxml.config import ModelConfig

    hf_config = _hf_model_config(model, "Kimi Linear")
    cfg = ModelConfig.from_hf(hf_config)
    if cfg.architecture != "kimi_linear":
        raise ValueError(
            "Expected Hugging Face Kimi Linear model_type='kimi_linear', " f"got {getattr(hf_config, 'model_type', None)!r}."
        )
    state_dict = _hf_model_state_dict(model, "Kimi Linear")
    np_dtype = _resolve_np_dtype(dtype)

    def g(key, target_dtype=np_dtype):
        try:
            value = state_dict[key]
        except KeyError as e:
            raise KeyError(f"Hugging Face Kimi Linear state dict is missing required tensor {key!r}.") from e
        return _state_value_to_numpy(value, target_dtype, key)

    num_mla_heads = cfg.num_heads
    qk_head_dim = cfg.qk_head_dim
    nope_head_dim = cfg.qk_nope_head_dim
    value_head_dim = cfg.v_head_dim
    num_kda_heads = cfg.kda_num_heads
    kda_head_dim = cfg.kda_head_dim
    conv_width = cfg.kda_short_conv_kernel_size

    def mla_attention_params(prefix):
        return {
            "q_proj": {
                "kernel": g(prefix + "q_proj.weight").T.reshape(-1, num_mla_heads, qk_head_dim),
            },
            "kv_a_proj_with_mqa": {"kernel": g(prefix + "kv_a_proj_with_mqa.weight").T},
            "kv_a_layernorm": {"weight": g(prefix + "kv_a_layernorm.weight")},
            "kv_b_proj": {
                "kernel": g(prefix + "kv_b_proj.weight").T.reshape(-1, num_mla_heads, nope_head_dim + value_head_dim),
            },
            "o_proj": {"kernel": g(prefix + "o_proj.weight").T},
        }

    def kda_attention_params(prefix):
        def projection(name):
            return {"kernel": g(prefix + name + ".weight").T.reshape(-1, num_kda_heads, kda_head_dim)}

        def convolution(name):
            weight = g(prefix + name + ".weight")
            expected = (num_kda_heads * kda_head_dim, 1, conv_width)
            if weight.shape != expected:
                raise ValueError(f"Hugging Face Kimi Linear {name} weight must have shape {expected}, got {weight.shape}.")
            return {"weight": weight.reshape(num_kda_heads, kda_head_dim, conv_width)}

        return {
            "A_log": g(prefix + "A_log", np.float32).reshape(num_kda_heads),
            "dt_bias": g(prefix + "dt_bias", np.float32).reshape(num_kda_heads, kda_head_dim),
            "q_proj": projection("q_proj"),
            "q_conv1d": convolution("q_conv1d"),
            "k_proj": projection("k_proj"),
            "k_conv1d": convolution("k_conv1d"),
            "v_proj": projection("v_proj"),
            "v_conv1d": convolution("v_conv1d"),
            "f_a_proj": {"kernel": g(prefix + "f_a_proj.weight").T},
            "f_b_proj": {
                "kernel": g(prefix + "f_b_proj.weight").T.reshape(kda_head_dim, num_kda_heads, kda_head_dim),
            },
            "b_proj": {"kernel": g(prefix + "b_proj.weight").T},
            "g_a_proj": {"kernel": g(prefix + "g_a_proj.weight").T},
            "g_b_proj": {
                "kernel": g(prefix + "g_b_proj.weight").T.reshape(kda_head_dim, num_kda_heads, kda_head_dim),
            },
            "o_norm": {"weight": g(prefix + "o_norm.weight")},
            "o_proj": {"kernel": g(prefix + "o_proj.weight").T},
        }

    def dense_mlp_params(prefix):
        return {
            "gate_proj": {"kernel": g(prefix + "gate_proj.weight").T},
            "up_proj": {"kernel": g(prefix + "up_proj.weight").T},
            "down_proj": {"kernel": g(prefix + "down_proj.weight").T},
        }

    def sparse_moe_params(prefix):
        gate_up_experts = []
        down_experts = []
        for expert_idx in range(cfg.n_routed_experts):
            expert_prefix = prefix + f"experts.{expert_idx}."
            gate = g(expert_prefix + "w1.weight")
            up = g(expert_prefix + "w3.weight")
            if gate.shape != up.shape:
                raise ValueError(
                    f"Hugging Face Kimi Linear expert {expert_idx} w1/w3 shapes must match, "
                    f"got {gate.shape} and {up.shape}."
                )
            gate_up_experts.append(np.concatenate((gate, up), axis=0))
            down_experts.append(g(expert_prefix + "w2.weight"))
        gate_up = np.stack(gate_up_experts, axis=0).transpose(0, 2, 1)
        down = np.stack(down_experts, axis=0).transpose(0, 2, 1)
        return {
            "gate_kernel": g(prefix + "gate.weight").T,
            "e_score_correction_bias": g(prefix + "gate.e_score_correction_bias"),
            "gate_up_kernel": gate_up,
            "down_kernel": down,
            "shared_gate_proj": {"kernel": g(prefix + "shared_experts.gate_proj.weight").T},
            "shared_up_proj": {"kernel": g(prefix + "shared_experts.up_proj.weight").T},
            "shared_down_proj": {"kernel": g(prefix + "shared_experts.down_proj.weight").T},
        }

    model_params = {"embed_tokens": {"embedding": g("model.embed_tokens.weight")}}
    for idx in range(cfg.num_layers):
        layer_prefix = f"model.layers.{idx}."
        layer_params = {
            "self_attn": (
                kda_attention_params(layer_prefix + "self_attn.")
                if cfg.is_kda_layer(idx)
                else mla_attention_params(layer_prefix + "self_attn.")
            ),
            "input_layernorm": {"weight": g(layer_prefix + "input_layernorm.weight")},
            "post_attention_layernorm": {"weight": g(layer_prefix + "post_attention_layernorm.weight")},
        }
        if cfg.is_moe_layer(idx):
            layer_params["block_sparse_moe"] = sparse_moe_params(layer_prefix + "block_sparse_moe.")
        else:
            layer_params["mlp"] = dense_mlp_params(layer_prefix + "mlp.")
        model_params[f"layers_{idx}"] = layer_params
    model_params["norm"] = {"weight": g("model.norm.weight")}
    return {"params": {"model": model_params, "lm_head": {"kernel": g("lm_head.weight").T}}}


HFArchitecture = Literal[
    "auto",
    "llama",
    "mistral",
    "deepseek",
    "deepseek_v3",
    "glm4_moe_lite",
    "glm4",
    "glm",
    "kimi_k2",
    "kimi_linear",
    "kimi",
    "gpt_neox",
    "neox",
    "gemma3",
    "gemma",
]


def _normalize_hf_architecture(architecture: str) -> str:
    if not isinstance(architecture, str):
        raise TypeError(f"architecture must be a string, got {type(architecture)}.")
    normalized = architecture.lower().replace("-", "_")
    match normalized:
        case "auto" | "llama" | "mistral" | "neox" | "gemma":
            return normalized
        case "deepseek" | "deepseek_v3":
            return "deepseek"
        case "glm4_moe_lite" | "glm4" | "glm":
            return "glm"
        case "kimi_k2":
            return "kimi_k2"
        case "kimi_linear" | "kimi":
            return "kimi_linear"
        case "gpt_neox":
            return "neox"
        case "gemma3" | "gemma_3":
            return "gemma"
        case _:
            raise ValueError(f"Unsupported Hugging Face architecture {architecture!r}.")


def _normalize_hf_model_name(name: str | PathLike[str]) -> str:
    try:
        normalized = fspath(name)
    except TypeError as e:
        raise TypeError(f"name must be a string or path-like object, got {type(name)}.") from e
    if not isinstance(normalized, str):
        raise TypeError(f"name must resolve to a string path, got {type(normalized)}.")
    if not normalized or not normalized.strip():
        raise ValueError("name must be non-empty.")
    return normalized


def _validate_hf_dtype(dtype: str):
    _resolve_np_dtype(dtype)
    return dtype


def _validate_hf_model_type(config, expected_model_types: tuple[str, ...], architecture_name: str):
    model_type = getattr(config, "model_type", None)
    if model_type not in expected_model_types:
        raise ValueError(
            f"Expected Hugging Face {architecture_name} model_type to be one of {expected_model_types}, got {model_type!r}."
        )


def _infer_hf_architecture(name: str, **config_kwargs) -> str:
    try:
        from transformers import PretrainedConfig
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    config_dict, _ = PretrainedConfig.get_config_dict(name, **config_kwargs)
    model_type = config_dict.get("model_type")
    match model_type:
        case "llama":
            return "llama"
        case "mistral":
            return "mistral"
        case "deepseek_v3":
            return "deepseek"
        case "glm4_moe_lite":
            return "glm"
        case "kimi_k2":
            return "kimi_k2"
        case "kimi_linear":
            return "kimi_linear"
        case "gpt_neox":
            return "neox"
        case "gemma3":
            return "gemma"
        case "gemma3_text":
            return "gemma"
        case _:
            raise ValueError(f"Unsupported Hugging Face model type {model_type!r}.")


def load_model_from_hf(
    name: str,
    architecture: HFArchitecture = "auto",
    dtype: str = "float32",
    **from_pretrained_kwargs,
):
    """Load a supported Hugging Face causal LM into a jaxml model and parameter tree."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    architecture = _normalize_hf_architecture(architecture)
    if architecture == "auto":
        architecture = _infer_hf_architecture(name, **from_pretrained_kwargs)

    match architecture:
        case "llama":
            return load_llama_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "mistral":
            return load_mistral_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "deepseek" | "deepseek_v3":
            return load_deepseek_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "glm" | "glm4" | "glm4_moe_lite":
            return load_glm_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "kimi_k2":
            return load_kimi_k2_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "kimi" | "kimi_linear":
            return load_kimi_linear_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "gpt_neox" | "neox":
            return load_neox_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case "gemma3" | "gemma":
            return load_gemma_from_hf(name, dtype=dtype, **from_pretrained_kwargs)
        case _:
            raise ValueError(f"Unsupported Hugging Face architecture {architecture!r}.")


def load_llama_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[LlamaModelWithHead, dict]:
    """Load Huggingface llama compatible models directly from either local path
    or the hf-hub identifier."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "Llama")
    _validate_hf_model_type(_config, ("llama",), "Llama")
    params = to_llama_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = LlamaModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_mistral_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[LlamaModelWithHead, dict]:
    """Load Huggingface Mistral compatible models. Mistral reuses the Llama model definition."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "Mistral")
    _validate_hf_model_type(_config, ("mistral",), "Mistral")
    params = to_mistral_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = LlamaModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def _load_mla_moe_from_hf(
    name: str,
    *,
    dtype: str,
    expected_model_types: tuple[str, ...],
    architecture_name: str,
    **from_pretrained_kwargs,
) -> tuple[DeepseekModelWithHead, dict]:
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, architecture_name)
    _validate_hf_model_type(_config, expected_model_types, architecture_name)
    params = to_deepseek_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = DeepseekModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_deepseek_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[DeepseekModelWithHead, dict]:
    """Load a Hugging Face DeepSeek-V3 / R1 model into a jaxml DeepseekModelWithHead."""
    return _load_mla_moe_from_hf(
        name,
        dtype=dtype,
        expected_model_types=("deepseek_v3",),
        architecture_name="DeepSeek",
        **from_pretrained_kwargs,
    )


def load_glm_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[DeepseekModelWithHead, dict]:
    """Load a Hugging Face GLM-4.7-Flash model through the shared MLA/MoE implementation."""
    return _load_mla_moe_from_hf(
        name,
        dtype=dtype,
        expected_model_types=("glm4_moe_lite",),
        architecture_name="GLM",
        **from_pretrained_kwargs,
    )


def load_kimi_k2_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[DeepseekModelWithHead, dict]:
    """Load a Hugging Face Kimi-K2 model through the shared MLA/MoE implementation."""
    return _load_mla_moe_from_hf(
        name,
        dtype=dtype,
        expected_model_types=("kimi_k2",),
        architecture_name="Kimi",
        **from_pretrained_kwargs,
    )


def load_kimi_linear_from_hf(
    name: str,
    dtype: str = "float32",
    **from_pretrained_kwargs,
) -> tuple[KimiLinearModelWithHead, dict]:
    """Load Kimi Linear directly from sharded safetensors.

    This deliberately avoids importing the official Torch model, whose remote
    implementation depends on the CUDA-oriented FLA package. Only one source
    checkpoint shard is resident while the converted JAX tree is assembled.
    """
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)

    from jaxml.config import ModelConfig

    hf_model = _load_kimi_linear_safetensors_source(name, **from_pretrained_kwargs)
    hf_config = _hf_model_config(hf_model, "Kimi Linear")
    _validate_hf_model_type(hf_config, ("kimi_linear",), "Kimi Linear")
    params = to_kimi_linear_jax_params(hf_model, dtype=dtype)
    config = ModelConfig.from_hf(hf_config)
    model = KimiLinearModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_kimi_from_hf(
    name: str,
    dtype: str = "float32",
    **from_pretrained_kwargs,
) -> tuple[KimiLinearModelWithHead, dict]:
    """Alias for :func:`load_kimi_linear_from_hf`."""
    return load_kimi_linear_from_hf(name, dtype=dtype, **from_pretrained_kwargs)


def load_neox_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[GPTNeoXModelWithHead, dict]:
    """Load Huggingface gpt-neox compatible models directly from either local path
    or the hf-hub identifier."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _config = _hf_model_config(_model, "GPT-NeoX")
    _validate_hf_model_type(_config, ("gpt_neox",), "GPT-NeoX")
    params = to_neox_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_config)
    model = GPTNeoXModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params


def load_gemma_from_hf(name: str, dtype: str = "float32", **from_pretrained_kwargs) -> tuple[GemmaModelWithHead, dict]:
    """Load Huggingface gemma3 compatible models directly from either local path
    or the hf-hub identifier."""
    name = _normalize_hf_model_name(name)
    dtype = _validate_hf_dtype(dtype)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError("Please install transformers library.") from e

    from jaxml.config import ModelConfig

    _wrapper = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
    _wrapper_config = _hf_model_config(_wrapper, "Gemma")
    _validate_hf_model_type(_wrapper_config, ("gemma3",), "Gemma")
    try:
        _model = _wrapper.language_model
    except AttributeError as e:
        raise ValueError("Expected Hugging Face Gemma model to expose a language_model module.") from e
    _model_config = _hf_model_config(_model, "Gemma text")
    _validate_hf_model_type(_model_config, ("gemma3_text",), "Gemma text")
    params = to_gemma_jax_params(_model, dtype=dtype)
    config = ModelConfig.from_hf(_model_config)
    model = GemmaModelWithHead(config, dtype=_resolve_np_dtype(dtype))
    return model, params
