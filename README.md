# jaxml

A relatively carefully written library in JAX to support my own research (and hopefully help others too).

I do not intend to make it a framework that satisfies every model type and everybody (unless it gets viral which is unlikely). But anybody is free to contribute (so far just myself though).

# Structure

It contains two things in a separate fashion:
1. Model architectures (`jaxml.models`)
2. Inference engine (`jaxml.inference_engine`)

Within the definition of model architectures, it also uses the following
1. Neural network components (`jaxml.nn`)
2. Model configs (`jaxml.config`)


# Support and features

Currently supported:
- [x] Llama
- [x] Mistral (full-attention configs; sliding-window not yet supported)
- [x] GPT-NeoX
- [x] Gemma3
- [x] DeepSeek-V3 / R1 (MLA + fine-grained MoE, including YaRN)
- [x] GLM-4.7-Flash (`glm4_moe_lite`, through the shared MLA/MoE implementation)
- [x] Kimi-K2 text architecture (DeepSeek-V3-compatible MLA/MoE + YaRN)
- [x] Kimi Linear (hybrid KDA/NoPE-MLA reference path; optimized TPU KDA prefill is still pending)
- [x] Qwen3.5 / Qwen3.6 MoE text architecture (hybrid Gated DeltaNet/full attention; no vision or MTP)

For Qwen, the canonical loader architecture is `qwen3_5`; accepted aliases include
`qwen3_5_moe`, `qwen3_5_moe_text`, `qwen35`, `qwen3_6`, `Qwen3.6`, `qwen36`, and
`qwen`. Both the outer `qwen3_5_moe` and text-only `qwen3_5_moe_text` Hugging Face
config types are recognized.

Inference engine features:
- [x] tensor parallel and data parallel (using JAX sharding semantics)
- [x] AOT-compile prefill and decode functions, and cache them!
- [ ] (pending a bug fix in JAX) Allow JAX-flash-attention (`jax.experimental.pallas.ops.flash_attention`)

# Installation

Install the base package for the JAX model and lower-level inference APIs:

```bash
pip install jaxml
```

Hugging Face checkpoint loading and text tokenization use optional PyTorch and Transformers dependencies:

```bash
pip install "jaxml[hf]"
```

Add the TPU runtime extra when running on TPU. From a source checkout, use `.` in place of `jaxml` in these commands.

```bash
pip install "jaxml[hf,tpu]"
```

# Quick start

```python
from jaxml import GenerationConfig, TextGenerationPipeline

pipeline = TextGenerationPipeline.from_hf(
    "EleutherAI/pythia-70m",
    architecture="neox",
    model_dtype="float32",
)

text = pipeline.generate_text(
    "To implement quick sort,",
    generation_config=GenerationConfig(max_new_tokens=64, temperature=0.0),
)
print(text)
```

`model_dtype` controls converted weights and model compute; the KV cache uses the
same dtype unless `engine_dtype` is explicitly overridden.

The lower-level engine still accepts token arrays directly through `jaxml.inference_engine.Engine`.

## TPU v4-8 guidance

A v4-8 is four TPU chips, so the intended starting topology for the MoE models is
`InferenceConfig(tp_size=4, dp_size=1, ep_size=1)`. The expert weights already
shard along tensor-parallel intermediate dimensions; expert-parallel collectives
are not required for the first useful single-host setup.

The two primary v4-8 optimization targets are GLM-4.7-Flash and
Qwen3.6-35B-A3B: both are useful modern MoE checkpoints that fit this single-host
topology more realistically than the largest Kimi releases. Kimi-K2 model/config
compatibility is included for completeness, but its trillion-parameter released
checkpoint is not a v4-8 deployment target.

Always load the 35B Qwen checkpoint as BF16 on v4-8 (the dedicated Qwen loader
defaults to BF16; the generic pipeline keeps its backward-compatible float32
default):

```python
from jaxml import InferenceConfig, TextGenerationPipeline

pipeline = TextGenerationPipeline.from_hf(
    "Qwen/Qwen3.6-35B-A3B",
    architecture="qwen3_6",
    model_dtype="bfloat16",
    inference_config=InferenceConfig(tp_size=4),
    use_tpu=True,
)
```

The text tree has about 34.66B parameters: roughly 64.6 GiB in BF16 versus
129.1 GiB in float32. The loader warns when a float32 Qwen tree alone would
exceed the aggregate 128 GiB HBM of v4-8.

DeepSeek-compatible MLA layers, including GLM and Kimi variants, persist only the
normalized `kv_lora` latent and rotary-key slice instead of expanded per-head K/V.
This reduces persistent cache HBM, but the current attention path still runs
`kv_b_proj` over the cached latent and may materialize expanded K/V transiently.
Qwen Gated DeltaNet currently uses the reference scan; a chunkwise TPU prefill
kernel is pending, and a populated cache currently continues one token at a time.
The recommended v4-8 path keeps `ep_size=1`; expert-sharded Megablox execution is
not yet validated. Qwen support is text-only and excludes vision and MTP paths.

Checkpoint conversion currently materializes the converted parameter tree on the
host before the engine shards it. The Kimi Linear and Qwen loaders read one
safetensors tensor at a time and avoid constructing the official Torch model, but
the converted tree still needs substantial host RAM. Direct safetensors-to-device
streaming is the next loader milestone.

# Development

Tests are split by cadence:

- `critical`: focused CPU push gate for public API, generation, cache, sampling, and validation coverage.
- `milestone`: full CPU suite for model internals, conversion utilities, dependency/tooling checks, and exhaustive regressions.
- `tpu`: TPU-only checks for the local TPU runtime and JAX/libtpu setup.

Current critical modules:

- `tests/test_cache.py`
- `tests/test_generate_core.py`
- selected runtime smoke cases in `tests/test_generation.py`
- `tests/test_public_api.py`
- `tests/test_sampling.py`
- `tests/test_text_generation.py`
- `tests/test_validation.py`

Current milestone modules:

- `tests/test_aot_cache.py`
- `tests/test_attention.py`
- `tests/test_config.py`
- `tests/test_deepseek.py`
- `tests/test_dependency_surface.py`
- `tests/test_experimental_rnn_discrete.py`
- `tests/test_gemma.py`
- `tests/test_generation.py`
- `tests/test_glm_kimi.py`
- `tests/test_hf_utils.py`
- `tests/test_kimi_linear.py`
- `tests/test_llama.py`
- `tests/test_model_heads.py`
- `tests/test_model_inputs.py`
- `tests/test_modules.py`
- `tests/test_neox.py`
- `tests/test_qwen3_5.py`
- `tests/test_qwen35_hf.py`
- `tests/test_rope.py`
- `tests/test_sharding.py` (multi-device cases need `XLA_FLAGS=--xla_force_host_platform_device_count=4`)

Run the push gate before every commit or push:

```bash
uv run --frozen --extra dev make verify-critical-cpu
```

This also builds the source distribution and wheel into `tmp/build-check` so packaging regressions fail the push gate.

Run the full CPU milestone suite before larger review points:

```bash
uv run --frozen --extra dev make verify-milestone-cpu
```

TPU validation is intentionally split out because it requires a visible TPU runtime and `libtpu`:

```bash
uv run --frozen --extra dev --extra tpu make verify-tpu
```

Before changing pinned dependencies, audit direct dependency drift across the base, HF, dev, and TPU dependency sets:

```bash
uv run --frozen --extra dev make dependency-drift
```

Probe candidate JAX/JAXLIB pins in a temporary copy before editing committed pins. Start with the critical CPU probe:

```bash
uv run --frozen --extra dev make dependency-upgrade-probe JAX_VERSION=0.10.2
```

For a runtime dependency upgrade milestone, also run the full CPU and TPU probes:

```bash
uv run --frozen --extra dev make dependency-upgrade-probe-milestone-cpu JAX_VERSION=0.10.2
uv run --frozen --extra dev make dependency-upgrade-probe-tpu JAX_VERSION=0.10.2
```

Set `JAXLIB_VERSION=...` as well when the candidate JAX/JAXLIB versions intentionally differ, and `LIBTPU_VERSION=...` for a candidate TPU runtime pin. The probes rewrite only `tmp/dependency-upgrade-probe`, refresh that copy's lockfile, and run the selected gate there.

Compiled AOT cache entries include JAX/JAXLIB runtime metadata; inspect an entry with `jaxml.utils.compiled_fn_metadata(...)` before reusing caches across dependency upgrades.

GitHub CI runs shared push checks (`lock-check`, `dependency-check`, `lint`, `format-check`, and `build-check`) once on Python 3.12, then runs critical CPU tests on Python 3.11 and 3.12. The full milestone CPU suite runs on the weekly scheduled workflow and manual dispatch. TPU tests are excluded from CPU suites and remain a local/manual gate.
