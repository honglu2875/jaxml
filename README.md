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
