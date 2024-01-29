import math
import torch
import torch.nn as nn

def position_embedding(embed_dim, max_len):
    '''
    @returns ret: (max_len, embed_dim)
    '''
    ret = torch.zeros(max_len, embed_dim)
    for pos in range(max_len):
        for j in range(0, embed_dim, 2):
            ret[pos][j] = math.sin(pos / pow(10000, j / embed_dim))
            ret[pos][j + 1] = math.cos(pos / pow(10000, j / embed_dim))
    ret.requires_grad = False
    return ret
