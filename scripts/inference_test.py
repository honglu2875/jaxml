import time
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from jaxml.utils import load_llama_from_hf
from jaxml.inference_engine.engine import Engine, InferenceConfig


model_name = "NousResearch/Meta-Llama-3-8B"
model, params = load_llama_from_hf(model_name)
#params = model.get_params(weights=params)
#params = jax.tree.map(lambda x: jnp.array(x), params)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = ["The weather of Chicago is", "To implement quick sort,"]
encoded = tokenizer(prompt, return_tensors='np')
prompt_tokens = jnp.array(encoded.input_ids)
attention_mask = jnp.array(encoded.attention_mask)

config = InferenceConfig(max_sequence_length=200, tp_size=1)
engine = Engine(model, config, params)
engine.init_params(use_tpu=True)

#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
start = time.perf_counter()
output = engine.generate(prompt_tokens, attention_mask=attention_mask, max_new_tokens=150, top_k=32, temperature=1.0, fuse_decoding=False)
print("Time", time.perf_counter() - start)
print(tokenizer.batch_decode(np.array(output)))
