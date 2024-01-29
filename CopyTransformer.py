import torch
import torch.nn as nn
import embedding
import pe

class CopyTransformer(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.transformer = nn.Transformer(embed_dim, nhead, 2, 2, 512,batch_first=True)
        self.embed = embedding.Embedding(512, embed_dim)
        self.pe = pe.PositionalEmbedding(512, embed_dim)
        self.predictor = nn.Linear(128, 10)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = self.get_key_padding_mask(src)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.embed(src)
        tgt = self.embed(tgt)
        # 给src和tgt的token增加位置信息
        src = self.pe(src)
        tgt = self.pe(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        # out = self.predictor(out)
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask