import torch
import torch.nn as nn
import math
from einops import rearrange


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))

        src2 = self.feed_forward(src)
        src = self.norm2(src + self.dropout(src2))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, cond, tgt_mask=None, cond_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        tgt2 = self.cross_attn(tgt, cond, cond, cond_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        tgt2 = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt



class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cond, cond_mask=None):
        for layer in self.layers:
            cond = layer(cond, cond_mask)
        return self.norm(cond)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, cond, tgt_mask=None, cond_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, cond, tgt_mask, cond_mask)
        return self.norm(tgt)


class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size,num_spectial,type_classes,poi_classes, src_vocab_size=100, d_model=256, num_layers=4, num_heads=4, dim_feedforward=1024, dropout=0.1, max_len=3000):
        super(Transformer, self).__init__()
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
        self.decoder = Decoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size-2) # mask is not as output
        self.d_model = d_model
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*4),
            SiLU(),
            nn.Linear(self.d_model*4, self.d_model),
        )
        # different token type special category and poi
        self.token_type_layer = nn.Embedding(3,self.d_model)
        self.input_projection = nn.Linear(self.d_model*2, self.d_model)
        self.num_spectial=num_spectial
        self.type_classes=type_classes
        self.poi_classes=poi_classes

    def forward(self, x, cond_emb, t, batch):

        diffusion_step_emb = self.time_embed(timestep_embedding(t, self.d_model))
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]

        x = self.positional_encoding(position_ids) + self.tgt_embedding(x) + diffusion_step_emb.unsqueeze(1).expand(-1, seq_length, -1)

        tgt_mask = (batch.category_mask + batch.poi_mask).bool()

        # different token type special category and poi
        token_type = batch.category_mask + batch.poi_mask * 2
        token_type_emb = self.token_type_layer(token_type)

        x = self.input_projection(torch.cat([x,token_type_emb],dim=-1))

        output = self.decoder(tgt=x, cond=cond_emb, tgt_mask=tgt_mask, cond_mask=batch.mask)
        output = self.output_layer(output)

        output = rearrange(output, 'b l v -> b v l')
        
        return output