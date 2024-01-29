import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        assert(embed_dim % n_heads == 0)
        super().__init__()
        self.heads = n_heads
        self.embed_dim = embed_dim
        self.single_dim = embed_dim // n_heads
        self.heads = n_heads
        self.K = nn.Linear(self.single_dim, self.single_dim)
        self.V = nn.Linear(self.single_dim, self.single_dim)
        self.Q = nn.Linear(self.single_dim, self.single_dim)

    def forward(self, query, key, value, mask=None):
        '''
        Args:
            @params query, key, value: tensor, an embedding vector with size (b, seq_len, embed_dim)

        Returns:
            tensor, a vector after attention with size (b, seq_len, embed_dim)
        '''
        batch_size = query.shape[0]
        query_seq_len = query.shape[1]
        value_seq_len = value.shape[1]
        # print(key.shape)
        single_dim = value.shape[-1] // self.heads
        # print(value.shape)
        query = query.view(batch_size, query_seq_len, self.heads, single_dim)
        key = key.view(batch_size, value_seq_len, self.heads, single_dim)
        value = value.view(batch_size, value_seq_len, self.heads, single_dim)

        ### shape of query, key, value: (b, heads, seq_len, single_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        ### shape of q, k, v: (b, heads, seq_len, single_dim)
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)

        k = k.transpose(-1, -2)                                     # k: (b, heads, single_dim, key_seq_len)
        output = torch.matmul(q, k) / math.sqrt(single_dim)         # output: (b, single_dim, query_seq_len, key_seq_len)
        if mask is not None:
            # print(output.shape)
            # print(mask.shape)
            output = output.masked_fill_(mask==0, float("-1e20"))
        output = F.softmax(output, dim=-1)
        # print(output.shape, v.shape)
        output = torch.matmul(output, v)                            # output: (b, heads, query_seq_len, single_dim)
        # print(f'output.shape = {output.transpose(1,2).shape}')     
        # print(f'batch_size = {batch_size}, query_seq_len = {query_seq_len}, single_dim = {single_dim}, heads = {self.heads}')
        output = output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, single_dim * self.heads)  # output: (b, seq_len, original_dimension)
        # print(f'output.shape = {output.shape}')
        # here I ignore the out layer with shape (nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) to see if the layer really affects the training process
        return output
