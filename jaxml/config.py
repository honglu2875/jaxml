from flax import struct
from typing import Optional



@struct.dataclass
class ModelConfig:
    head_dim: int     
    num_heads: int
    num_layers: int
    max_position_embeddings: int
    vocab_size: int
    num_kv_heads: Optional[int] = None
    sliding_window: Optional[int] = None
    
    @property
    def num_key_value_heads(self):
        if self.num_kv_heads is None:
            return self.num_heads
        return self.num_kv_heads
    
    @property
    def hidden_size(self):
        return self.head_dim * self.num_heads