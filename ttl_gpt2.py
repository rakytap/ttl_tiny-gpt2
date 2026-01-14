from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel

import numpy as np
from typing import Optional


class GPT2ModelTTL(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        pass


class GPT2LMHeadModelTTL(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)

        self.ttl_transformer = GPT2ModelTTL(config)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        pass

    def compile_ttl(
        self,
        input_ids: Optional[np.ndarry] = None,
        attention_mask: Optional[np.ndarray] = None,
    ):

        pass
