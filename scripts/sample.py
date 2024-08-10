
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from jaxml.utils import load_llama_from_hf


model_name = "NousResearch/Meta-Llama-3-8B"
model, params = load_llama_from_hf(model_name)
params = model.get_params(weights=params)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = ["The weather of Chicago is", "To implement quick sort,"]
encoded = tokenizer(prompt, return_tensors='np')
prompt_tokens = jnp.array(encoded.input_ids)
attention_mask = jnp.array(encoded.attention_mask)


output = model.generate(params, prompt_tokens, attention_mask=attention_mask, max_new_tokens=100, do_sample=True, temperature=1.0, show_progress=True, no_jit=False)
print(tokenizer.batch_decode(np.array(output)))
