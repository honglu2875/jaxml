# jaxml

A relatively carefully written library in JAX to support my own research (and hopefully help otherstoo).

I do not intend to make it a framework that satisfies every model type and everybody (unless it gets viral which is unlikely). But anybody is free to contribute (so far just myself though).

# Structure

It contains two things in a separate fashion:
1. Model architectures (`jaxml.models`)
2. Inference engine (`jaxml.inference_engine`)

Within the definition of model architectures, it also uses the following
1. Neural network components (`jaxml.nn`)
2. Model configs (`jaxml.config`)


# Support and features

Currently support:
- [x] Llama

Inference engine features:
- [x] tensor parallel and data parallel (using JAX sharding semantics)
- [x] AOT-compile for prefilling function and decoding function, and cache them!
- [x] Allow JAX-flash-attention (`jax.experimental.pallas.ops.flash_attention`)

