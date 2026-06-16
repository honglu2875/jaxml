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
- [x] GPT-NeoX
- [x] Gemma3

Inference engine features:
- [x] tensor parallel and data parallel (using JAX sharding semantics)
- [x] AOT-compile prefill and decode functions, and cache them!
- [ ] (pending a bug fix in JAX) Allow JAX-flash-attention (`jax.experimental.pallas.ops.flash_attention`)

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

The lower-level engine still accepts token arrays directly through `jaxml.inference_engine.Engine`.

# Development

Tests are split by cadence:

- `critical`: required CPU push gate for broad API, generation, cache, and dependency-surface coverage.
- `milestone`: full CPU suite for model internals, conversion utilities, and broader regression coverage.
- `tpu`: TPU-only checks for the local TPU runtime and JAX/libtpu setup.

Current critical modules:

- `tests/test_cache.py`
- `tests/test_dependency_surface.py`
- `tests/test_generate_core.py`
- `tests/test_generation.py`
- `tests/test_model_inputs.py`
- `tests/test_public_api.py`
- `tests/test_sampling.py`
- `tests/test_text_generation.py`

Current milestone modules:

- `tests/test_aot_cache.py`
- `tests/test_attention.py`
- `tests/test_config.py`
- `tests/test_experimental_rnn_discrete.py`
- `tests/test_gemma.py`
- `tests/test_hf_utils.py`
- `tests/test_llama.py`
- `tests/test_model_heads.py`
- `tests/test_modules.py`
- `tests/test_neox.py`
- `tests/test_rope.py`

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

Before changing pinned dependencies, audit direct dependency drift across dev and TPU extras:

```bash
uv run --frozen --extra dev make dependency-drift
```

Compiled AOT cache entries include JAX/JAXLIB runtime metadata; inspect an entry with `jaxml.utils.compiled_fn_metadata(...)` before reusing caches across dependency upgrades.

GitHub CI runs shared push checks (`lock-check`, `dependency-check`, `lint`, `format-check`, and `build-check`) once on Python 3.12, then runs critical CPU tests on Python 3.11 and 3.12. The full milestone CPU suite runs on the weekly scheduled workflow and manual dispatch. TPU tests are excluded from CPU suites and remain a local/manual gate.
