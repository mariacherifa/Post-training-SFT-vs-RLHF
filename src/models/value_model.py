######################################################################
#                              Value model
######################################################################

import torch 
import torch.nn as nn 
from tinygpt.transformer import BlockTransformer

class ValueModel(nn.Module): 
    def __init__(self, n_embd, num_heads, block_size,vocab_size, dropout, num_layer):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd, bias= True)
        self.token_embedding_pos = nn.Embedding(block_size, n_embd, bias= True)
        self.blocks = nn.Sequential(*[BlockTransformer(n_embd, num_heads, block_size, dropout) for _ in range(num_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.value_head = nn.Linear(n_embd, 1)

    def forward(self, ids):

        B, T = ids.shape
        tok_embd = self.token_embedding(ids)
        pos = torch.arange(T, device = ids.device)
        pos_embd = self.token_embedding_pos(pos)

        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.lnf(x)  

        values = self.value_head(x)

        return values.squeeze(-1)






