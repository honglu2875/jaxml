import time
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from jaxml.hf_utils import load_llama_from_hf, load_neox_from_hf
from jaxml.inference_engine.engine import Engine, InferenceConfig


#model_name = "NousResearch/Meta-Llama-3-8B"
model_name = "EleutherAI/pythia-1b"
#model, params = load_llama_from_hf(model_name)
model, params = load_neox_from_hf(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = ["The weather of Chicago is", "To implement quick sort,"]

# ---- reference ---- #
with torch.no_grad():
    tok_input = tokenizer(prompt, return_tensors="pt")
    out = hf_model.generate(**tok_input, max_new_tokens=100, do_sample=True, temperature=1.0)
print(tokenizer.batch_decode(out))

encoded = tokenizer(prompt, return_tensors='np')
print(encoded)
prompt_tokens = jnp.array(encoded.input_ids)
attention_mask = jnp.array(encoded.attention_mask)

config = InferenceConfig(max_sequence_length=200, tp_size=1)
engine = Engine(model, config, params, dtype=jnp.float32)
engine.init_params(use_tpu=True)

#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
start = time.perf_counter()
output = engine.generate(prompt_tokens, attention_mask=attention_mask, max_new_tokens=150, top_k=0, temperature=1.0, fuse_decoding=True)
print("Time", time.perf_counter() - start)
print(tokenizer.batch_decode(np.array(output)))

# --- debug --- #
print("--- debug ---")
inp = encoded["input_ids"]
print(f"{inp=}")
print(f"{tok_input=}")
init_params = model.init(jax.random.PRNGKey(0), inp, output_attentions=True, output_hidden_states=True)
fwd = model.apply(init_params | params, inp, output_attentions=True, output_hidden_states=True)
fwd2 = hf_model(**tok_input, output_attentions=True, output_hidden_states=True)
diff_h = [np.abs(a - b.detach().numpy()) for a, b in zip(fwd.hidden_states, fwd2.hidden_states)]
diff_a = [np.abs(a - b.detach().numpy()) for a, b in zip(fwd.attention_weights, fwd2.attentions)]
breakpoint()
