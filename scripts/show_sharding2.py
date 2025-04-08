import jax
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from transformers import AutoTokenizer
from jaxml.hf_utils import load_llama_from_hf, load_neox_from_hf
from jaxml.inference_engine.engine import Engine, InferenceConfig
from jaxml.models.llama import LlamaDecoder

model_name = "NousResearch/Meta-Llama-3-8B"
model, params = load_llama_from_hf(model_name)


config = InferenceConfig(tp_size=4)
engine = Engine(model, config, params)
engine.init_params(use_tpu=True)

model = engine.model.bind(engine.params)

q_proj = model.model.layers[0].self_attn.q_proj
#q_proj_params = engine.params['params']['model']['layers_0']['self_attn']['q_proj']

key = jax.random.PRNGKey(0)
#mesh = Mesh(np.array(jax.devices()).reshape(1, 1, 4, 1), ('x', 'y', 'z', 'v'))
mesh = Mesh(
    devices=mesh_utils.create_device_mesh((1, 4)),
    axis_names=("data", "model"),
)
#spec = P('x', 'y', 'v')
spec = P('data', None, None)
named_sharding = jax.sharding.NamedSharding(mesh, spec)
input = jax.random.uniform(key, (4, 100, engine.model.config.hidden_size))
input = jax.device_put(input, named_sharding)
jax.debug.visualize_array_sharding(input[0])
breakpoint()
jax.debug.visualize_array_sharding(q_proj(input)[0, 0])


gate = engine.model.layers[0].mlp.gate_proj
gate_params = engine.params['params']['model']['layers_0']['mlp']['gate_proj']
