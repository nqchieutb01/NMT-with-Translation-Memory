import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask, Embedding,LearnedPositionalEmbedding
from config import FINE_TUNE

def layer_norm(x, variance_epsilon=1e-12):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + variance_epsilon)
    return x

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, sum=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if sum:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class MonoEncoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout):
        super(MonoEncoder, self).__init__()
        self.vocab = vocab
        self.src_embed = Embedding(vocab.size, embed_dim, vocab.padding_idx)
#         if FINE_TUNE:
#             self.src_pos_embed = LearnedPositionalEmbedding(embed_dim)
#             self.encoder_layernorm_embedding = nn.LayerNorm(embed_dim)
#             self.encoder_layer_norm = nn.LayerNorm(embed_dim)
#         else:
        self.src_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
#         self.src_pos_embed = LearnedPositionalEmbedding(embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.transformer = Transformer(layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.dropout = dropout


    def forward(self, input_ids):
        src_repr = self.embed_scale * self.src_embed(input_ids) + self.src_pos_embed(input_ids)
#         if FINE_TUNE:
#             src_repr = self.encoder_layernorm_embedding(src_repr)
        src_repr = F.dropout(src_repr, p=self.dropout, training=self.training)
        src_mask = torch.eq(input_ids, self.vocab.padding_idx)
        src_repr = self.transformer(src_repr, self_padding_mask=src_mask)
#         if FINE_TUNE:
#             src_repr = self.encoder_layer_norm(src_repr)
        return src_repr, src_mask
