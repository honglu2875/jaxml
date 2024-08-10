# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import linen as nn
from flax.linen.partitioning import with_sharding_constraint
import warnings

from ..cache import KVCache
from ..nn.attention import Attention, AttentionWithRoPE
from ..nn.embedding import Embed
from ..nn.linear import DenseGeneral
from ..nn.module import Block
from ..nn.norms import RMSNorm
from ..nn.position import RotaryEmbedding
from ..utils import get_default_pos_ids
from ..outputs import AttentionOutput, DecoderOutput, BaseModelOutputWithCache, CausalLMOutputWithCache


class LlamaMLP(Block):
    kernel_init: Any = nn.initializers.xavier_uniform
    act_fn: Any = jax.nn.silu

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for MLP.")
        # input dim supposed to be self.hidden_size
        self.gate_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="gate_proj",
            use_bias=self.use_bias,
        )
        self.up_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="up_proj",
            use_bias=self.use_bias,
        )
        self.down_proj = DenseGeneral(
            features=self.hidden_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("intermediate", "embed"),
            name="down_proj",
            use_bias=self.use_bias,
        )

    def __call__(self, x, **kwargs):
        assert (
            x.shape[-1] == self.hidden_size
        ), f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        gate = self.act_fn(self.gate_proj(x))
        proj = self.up_proj(x)
        x = self.down_proj(gate * proj)
        return x


class LlamaDecoder(Block):
    def setup(self):
        if not self.config.use_rope:
            self.self_attn = Attention(self.config, dtype=self.dtype)
        else:
            self.self_attn = AttentionWithRoPE(self.config, dtype=self.dtype)
        self.mlp = LlamaMLP(self.config, dtype=self.dtype)
        self.input_layernorm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=self.norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=self.norm_eps,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> DecoderOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
        )

        hidden_states = residual + output.attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual

        return DecoderOutput(
            hidden_states=hidden_states,
            kv_cache=output.kv_cache,
            attention_weight=output.attention_weight,
        )


class LlamaModel(Block):
    def setup(self):
        self.embed_tokens = Embed(
            num_embeddings=self.config.vocab_size, 
            features=self.hidden_size, 
            dtype=self.dtype
        )
        self.layers = [LlamaDecoder(self.config, dtype=self.dtype) for _ in range(self.num_layers)]
        self.norm = RMSNorm(self.hidden_size, eps=self.norm_eps)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[list[KVCache]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> BaseModelOutputWithCache:
        batch_size, seq_length = input_ids.shape

        if attention_mask is None:
            # need to apply a default value if kv_cache is either unused or empty
            if kv_caches is None or kv_caches[0].mask is None:
                # our convention is that negative ids (such as -100) is masked by default.
                attention_mask = input_ids >= 0


        inputs_embeds = self.embed_tokens(input_ids).astype(self.dtype)
        hidden_states = with_sharding_constraint(
            inputs_embeds, ("batch", "length", "embed")
        )

        all_hidden_states = []
        all_self_attns = []
        next_kv_caches = []

        for idx, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            kv_cache = None if kv_caches is None else kv_caches[idx]

            output = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )
            hidden_states, kv_cache, attn_weight = (
                output.hidden_states,
                output.kv_cache,
                output.attention_weight
            )


            all_self_attns.append(attn_weight)
            next_kv_caches.append(kv_cache)


        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        return BaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            kv_caches=tuple(next_kv_caches) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attention_weights=tuple(all_self_attns) if output_attentions else None,
        )        
        
        
class LlamaForCausalLM(Block):
    
    lm_head_init: Any = nn.initializers.xavier_uniform
    lm_head_init_args: tuple = ()

    def init_cache(self, max_seq_len) -> list[KVCache]:
        num_layers = self.config.num_layers
        return [KVCache.init(max_seq_len, None, None, dtype=self.dtype) for _ in range(num_layers)]

    @staticmethod
    def mesh_sharding(pspec: Optional[PartitionSpec], mesh: Optional[Mesh]) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    def _shard_params(self, x, y: PartitionSpec):
        if x.ndim != len(y.spec):
            assert (
                x.ndim == 2 and len(y.spec) == 3
            ), f"The shape of x ({x.shape}) and the sharding spec ({y.spec}) does not match"
            warnings.warn(
                f"The parameter has 2 axis ({x.shape}) while the sharding spec ({y.spec}) has 3 axis. "
                "Attempting to reshape into [:, :, head_dim], but please confirm that this is the intended behavior."
            )
            return jax.device_put(
                x.reshape(
                    (
                        x.shape[0],
                        -1,
                        self.head_dim,
                    )
                ),
                y,
            )
        return jax.device_put(x, y)


    def get_params(self, tp_size: int = 4, weights: Any = None, sharded: bool = True):
        """
        Get the properly sharded parameters.
        Args:
            tp_size: the tensor-parallel size.
            weights: whether a tree of weights are already given (but may not be sharded).
        Returns:
            a tree of properly sharded parameters
        """
        if sharded and not (len(jax.devices()) > 1 and len(jax.devices()) % tp_size == 0):
            sharded = False
            warning.warn(f"Cannot shard across devices: {jax.devices()}.")

        key = jax.random.PRNGKey(0)

        # (dp, tp)
        mesh_layout = (len(jax.devices()) // tp_size, tp_size)

        dummy_input = jnp.array(
            [[1 for _ in range(mesh_layout[1])] for _ in range(mesh_layout[0])]
        )
        abstract_variables = jax.eval_shape(self.init, key, dummy_input)
        rules = (
            ("batch", "data"),
            ("heads", "model"),
            ("kv_length", None),
            ("length", None),
            ("intermediate", "model"),
            ("heads_merged", "model"),
            ("head_states", None),
        )

        if sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(mesh_layout),
                axis_names=("data", "model"),
            )

            logical_state_spec = nn.get_partition_spec(abstract_variables)
            logical_state_sharding = nn.logical_to_mesh_sharding(
                logical_state_spec, mesh, rules
            )

            input_sharding = self.mesh_sharding(
                PartitionSpec("data", None), mesh
            )  # dimensions: (batch, length)

            if weights is not None:
                assert isinstance(
                    weights, dict
                ), f"weights must be a dict, got {type(weights)}"
                assert (
                    "params" in weights
                ), f"The key params not found in 'weights'. Got {weights.keys()}"

                if sharded:
                    params = {
                        "params": jax.tree_util.tree_map(
                            self._shard_params,
                            weights["params"],
                            logical_state_sharding["params"],
                        )
                    }
                else:
                    params = weights
            else:
                params = jax.jit(
                    self.init,
                    in_shardings=(
                        self.mesh_sharding(None, mesh),
                        input_sharding,
                    ),  # PRNG key and x
                    out_shardings=logical_state_sharding,
                )(key, dummy_input)
        else:
            params = self.init(key, dummy_input)

        return params

    def prepare_input(self, inputs, dp_size: int = 1, dtype: Any = None):
        mesh = Mesh(
            devices=mesh_utils.create_device_mesh(
                (dp_size, jax.device_count() // dp_size)
            ),
            axis_names=("data", "model"),
        )
        inputs = jax.device_put(
            inputs, self.mesh_sharding(PartitionSpec("data", None), mesh)
        )
        if dtype is not None:
            inputs = jax.tree_util.tree_map(lambda x: x.astype(dtype), inputs)
        return inputs

    def setup(self):
        self.model = LlamaModel(
            self.config, dtype=self.dtype
        )
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=self.lm_head_init,
            kernel_init_args=self.lm_head_init_args,
            with_logical_partitioning=True,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[list[KVCache]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithCache:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithCache(
            logits=logits,
            kv_caches=outputs.kv_caches,
            hidden_states=outputs.hidden_states,
            attention_weights=outputs.attention_weights,
        )

    def wrapped_apply_fn(
        self,
        params,
        input_token,
        attention_mask=None,
        kv_caches=None,
        use_cache=True,
    ) -> tuple[jnp.ndarray, list[KVCache]]:

        out, _ = self.apply(
            params,
            input_token,
            position_ids=None,
            attention_mask=attention_mask,
            mutable=("cache",),
            #output_hidden_states=False, # maybe allow for toggling of hidden states in the future
            #output_attentions=False, # maybe allow for toggling of attn wts in the future
            kv_caches=kv_caches,
            use_cache=use_cache,
        )  # return a tuple (CausalLMOutputWithCache, dict) where dict is the mutable cache

        return out.logits, out.kv_caches

    def generate(
        self,
        params,
        prompt_tokens: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        do_sample: bool = True,
        seed: int = 0,
        max_new_tokens: int = 10,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        no_jit: bool = False,
        show_progress: bool = False,
    ):
        if no_jit:
            apply = self.wrapped_apply_fn
        else:
            apply = jax.jit(self.wrapped_apply_fn, static_argnames=("use_cache",))

        kv_caches = self.init_cache(max_seq_len=prompt_tokens.shape[1] + max_new_tokens)
        
        from .._generate import generate

        return generate(
            params,
            apply,
            prompt_tokens,
            attention_mask,
            kv_caches,
            do_sample=do_sample,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            show_progress=show_progress,
        )

