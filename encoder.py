import torch
import torch.nn as nn
import embedding
import pe
import attention
import math

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor, n_heads):
        super().__init__()
        assert(embed_dim % n_heads == 0)
        self.embed_dim = embed_dim
        self.factor = expansion_factor
        self.heads = n_heads
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.attention = attention.MultiHeadAttention(embed_dim, n_heads)
        

    def forward(self, q, k, v):
        '''
        Args:
            @params x: tensor, (b, seq_len, embed_dim)

        Return:
            tensor, (b, seq_len, embed_dim)
        '''
        ### multi-head attention + add & norm layer
        # print(f'v.shape = {v.shape}')
        # print(f'temp.shape = {temp.shape}')
        output = q + self.attention(q, k, v)
        output = self.ln1(output)
        # output = self.dropout1(output)

        ### ffn + add & norm layer
        output = output + self.ffn(output)
        output = self.ln2(output)
        # output = self.dropout2(output)
        return output


    


class Encoder(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, num_layers, expansion_factor, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        # print(embed_dim, src_vocab_size)
        self.vocab_size = src_vocab_size
        self.enc_embed = embedding.embed(src_vocab_size, embed_dim)
        # self.pos_embed = pe.position_embedding(embed_dim, src_vocab_size)
        self.blocks = num_layers
        self.factor = expansion_factor
        self.heads = n_heads
        self.network = nn.ModuleList([EncoderBlock(embed_dim, expansion_factor, n_heads) for i in range(self.blocks)])
        # self.pos_embed = pe.position_embedding(self.embed_dim, 512)                 # Note: 512 is an arbitrary number which limits the seq_len upbound
        self.pos_embed = pe.position_embedding(512, embed_dim)

    def forward(self, x):
        '''
        x: torch.tensor (b, seq_len)
        out: torch.tensor (b, seq_len, embed_dim)
        '''
        seq_len = x.shape[1]
        # print(f'seq_len = {seq_len}')
        # print(f'x = {x}')
        # print(f'self.pos_embed.shape = {self.pos_embed.shape}')
        # print(f'enc_embed.shape = {self.enc_embed.}')
        # print(f'vocab_size = {self.vocab_size}')
        # out = self.enc_embed(x) + self.pos_embed[:seq_len, :]        # out: (b, seq, embed_dim)
        
        out = self.enc_embed(x) + self.pos_embed[:seq_len, :]
        for layer in self.network:
            out = layer(out, out, out)
        return out
