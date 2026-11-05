######################################################################
#                              Reward Models 
######################################################################


import torch 
import torch.nn  as nn 
from tinygpt.transformer import BlockTransformer


class RewardModel(nn.Module): 
    def __init__(self, n_embd, num_heads, block_size,vocab_size, dropout, num_layer):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[BlockTransformer(n_embd, num_heads, block_size, dropout) for _ in range(num_layer)])

        self.ln = nn.LayerNorm(n_embd)
        self.reward_head = nn.Linear(n_embd, 1)

    def forward(self,ids, attention_mask): 
        # attention mask: masks padded parts in a sequence 
        
        B,T = ids.shape 
        token_emb = self.embedding(ids)
        pos = torch.arange(T,  device=ids.device)
        token_pos = self.pos_embedding(pos)

        x = token_emb + token_pos
        x = self.blocks(x)
        x = self.ln(x)

        length_of_sequences = attention_mask.long().sum(dim = 1)

        if (length_of_sequences==0).any(): 
            raise ValueError("must be no empty sequence")

        Batch_index = torch.arange(B, device=ids.device)
        last_indices =  length_of_sequences  - 1
    
        last_hidden_representation = x[Batch_index, last_indices,: ]

   
        reward = self.reward_head(last_hidden_representation).squeeze(-1)

        return reward



        

