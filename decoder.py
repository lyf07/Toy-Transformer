import torch
import torch.nn as nn
import torch.nn.functional as F
import embedding
import pe
import attention
import encoder


class DecoderBlock(nn.Module):
    def __init__(self, dim, expansion_factor, n_heads):
        super().__init__()
        self.dim = dim
        self.heads = n_heads
        self.factor = expansion_factor
        self.ln = nn.LayerNorm(self.dim)
        self.mask_attention = attention.MultiHeadAttention(dim, n_heads)
        self.block = encoder.EncoderBlock(dim, expansion_factor, n_heads)
        # self.dp = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask=None):
        '''
        @params q, k, v: (b, seq_len, embed_dim)
        '''
        output = x + self.mask_attention(x, x, x, mask)
        # output = self.dp(self.ln(output))
        output = self.ln(output)
        output = self.block(output, enc_out, enc_out)
        return output

class Decoder(nn.Module):
    def __init__(self, embed_dim, tgt_vocab_size, num_layers, expansion_factor, n_heads):
        super().__init__()
        self.enc_embed = embedding.embed(tgt_vocab_size, embed_dim)
        self.dim = embed_dim
        self.num = num_layers
        self.factor = expansion_factor
        self.heads = n_heads
        self.pos_embed = pe.position_embedding(self.dim, 512)                 # Note: 512 is an arbitrary number which limits the seq_len upbound
        self.layers = nn.ModuleList([DecoderBlock(self.dim, self.factor, self.heads) for i in range(self.num)])
        self.linear = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, enc_out, dec_in, mask):
        '''
        @params enc_out(b, seq_len, embed_dim)
        @params dec_in (b, seq_len)
        @params mask (b, seq_len, seq_len)

        @returns output (b, tgt_vocab)
        '''
        seq_len = dec_in.shape[1]
        output = self.enc_embed(dec_in) + self.pos_embed[:seq_len, :]               # output: (b, seq_len, embed_dim)
        # TODO: ignore the dropout layer here to see if it really affects the training process
        
        for layer in self.layers:
            output = layer(output, enc_out, mask)
        
        output = self.linear(output)            # output: (b, seq_len, tgt_vocab)
        output = F.softmax(output, dim=1)
        # output = output.argmax(dim = 1)
        # print(self.enc_embed(torch.tensor([0])))
        return output
