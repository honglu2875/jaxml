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

import torch


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DummyPosEmb(torch.nn.Module):
    """Dummy positional embedding module to replace RotaryEmbedding."""

    def __init__(self):
        super(DummyPosEmb, self).__init__()

    def forward(self, x, *args, **kwargs):
        return (
            torch.ones(
                (1, x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            ),
            torch.zeros(
                (1, x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            ),
        )
