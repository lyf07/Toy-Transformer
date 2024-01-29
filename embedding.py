import torch
import torch.nn as nn

def embed(vocab_size, embed_dim):
    return nn.Embedding(vocab_size, embed_dim)

