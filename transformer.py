import torch
import torch.nn as nn
import encoder
import decoder


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, tgt_vocab_size, num_layers, expansion_factor, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.heads = n_heads
        self.encoder = encoder.Encoder(embed_dim, src_vocab_size, num_layers, expansion_factor, n_heads)
        self.decoder = decoder.Decoder(embed_dim, tgt_vocab_size, num_layers, expansion_factor, n_heads)

    def make_mask(self, tgt):
        '''
        @params tgt: tensor with shape (b, seq_len)
        @returns mask: a masked mask
        '''
        batch_size, seq_len = tgt.shape
        output = torch.ones(seq_len, seq_len)
        # in attention.py line 47, the output shape is output: (b, single_dim, query_seq_len, seq_len), so we need to expand it into (batch_size, 1, seq_len, seq_len) 
        output = torch.tril(output).expand(batch_size, 1, seq_len, seq_len) 
        return output

    def forward(self, src, tgt):
        '''
        specifically for machine translation
        Args: src, tgt
        Output: expect tgt target
        '''
        tgt_mask = self.make_mask(tgt)
        enc_out = self.encoder(src)
        output = self.decoder(enc_out, tgt, tgt_mask)
        return output
