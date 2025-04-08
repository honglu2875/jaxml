import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from jaxml.hf_utils import load_llama_from_hf, load_neox_from_hf
from jaxml.inference_engine.engine import Engine, InferenceConfig


model_name = "NousResearch/Meta-Llama-3-8B"
model, params = load_llama_from_hf(model_name)


config = InferenceConfig(tp_size=4)
engine = Engine(model, config, params)
engine.init_params(use_tpu=True)


jax.debug.visualize_array_sharding(engine.params['params']['model']['embed_tokens']['embedding'])
print("q, k, v param shapes:", engine.params['params']['model']['layers_0']['self_attn']['q_proj']['kernel'].shape)
print("q, k, v sharding (fixing embedding):")
jax.debug.visualize_array_sharding(engine.params['params']['model']['layers_0']['self_attn']['q_proj']['kernel'][0,:,:])
print("After attention, it goes through mlp: gate_proj(x) * up_proj(x) ---down_proj---> output.")
print("mlp gate_proj sharding:")
jax.debug.visualize_array_sharding(engine.params['params']['model']['layers_0']['mlp']['gate_proj']['kernel'])
print("mlp up_proj sharding:")
jax.debug.visualize_array_sharding(engine.params['params']['model']['layers_0']['mlp']['up_proj']['kernel'])
print("mlp down_proj sharding:")
jax.debug.visualize_array_sharding(engine.params['params']['model']['layers_0']['mlp']['down_proj']['kernel'])
print("It rinses and repeat over several layers until it reaches the end, lm_head.")
print("lm_head sharding:")
jax.debug.visualize_array_sharding(engine.params['params']['lm_head']['kernel'])
