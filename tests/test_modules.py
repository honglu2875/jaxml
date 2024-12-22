#  Copyright 2024 Honglu Fan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest

from jaxml.utils import torch_to_jax_states


@pytest.mark.parametrize("name", ["dense", "rms_norm", "layer_norm"])
def test_modules(jax_component_factory, torch_component_factory, name):
    jax_comp = jax_component_factory(name)
    torch_comp = torch_component_factory(name)

    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (4, 10, 48), dtype=jnp.float32)
        params = torch_to_jax_states(torch_comp, dtype=torch.float32)
        y = jax_comp.apply(params, x)
        with torch.no_grad():
            y2 = torch_comp(torch.tensor(np.array(x))).numpy()

        assert np.allclose(y, y2, atol=1e-5)
