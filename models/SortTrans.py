import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_token, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_token % num_heads == 0, "d_token is not divisible by num_heads"
        self.d_k = d_token // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_token, d_token)
        self.k_linear = nn.Linear(d_token, d_token)
        self.v_linear = nn.Linear(d_token, d_token)
        self.out_linear = nn.Linear(d_token, d_token)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attention = torch.softmax(scores, dim=-1)

        x = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)
        return self.out_linear(x)

class FeedForward(nn.Module):
    def __init__(self, d_token, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_token, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_token)

    def forward(self, x):
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class EncoderLayer(nn.Module):
    def __init__(self, d_token, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_token, num_heads)
        self.feed_forward = FeedForward(d_token, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x,):
        self_atten = self.self_attn(x, x, x)
        x = self_atten + self.dropout1(x)
        x = self.norm1(x)

        ff = self.feed_forward(x)
        x = ff + self.dropout2(x)
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_token, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_token, num_heads)
        self.feed_forward = FeedForward(d_token, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.norm3 = nn.LayerNorm(d_token)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None):
        
        # self attention
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x)

        # corss attention
        x = self.self_attn(x, enc_output, enc_output)
        x = self.dropout2(x)
        x = self.norm2(x)

        # fully connected
        ff_output = self.feed_forward(x)
        x = self.dropout3(ff_output)
        x = self.norm3(x)

        return ff_output

class seq_embedding(nn.Module):
    def __init__(self, d_token):
        super(seq_embedding, self).__init__()
        self.d_token = d_token
        self.dic = nn.Parameter(torch.randn(100, d_token))

    def forward(self, x):
        return self.dic[x-1]

class seq_transformer(nn.Module):
    def __init__(self, config):
        super(seq_transformer, self).__init__()
        self.d_token = config["d_token"]
        self.num_heads = config["num_heads"]
        self.d_ff = config["d_ff"]
        self.num_layers = config["num_layers"] 
        self.dropout = config["dropout"]
        # self.embedding = nn.Linear(1, self.d_token)
        # self.embedding = nn.Conv1d(1, self.d_token, 1)
        self.embedding = seq_embedding(self.d_token)
        self.position = PositionalEncoding(self.d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_token))
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_token, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.d_token, 10)
    
    def forward(self, x):
        # tokenlize
        # enc_output = self.embedding(x.unsqueeze(-1).permute(0, 2, 1))
        # enc_output = enc_output.permute(0, 2, 1)
        enc_output = self.embedding(x.int())

        # add the positional encoding
        # enc_output = enc_output  + self.position(enc_output)

        # add the cls token to pridict the right sequence
        # cls_token = nn.Parameter(torch.randn(1, 1, enc_output.shape[-1]).to(enc_output.device))
        cls_token_expended = self.cls_token.expand(x.shape[0], -1, -1)
        # print(enc_output.shape)
        # print(cls_token_expended.shape)
        enc_output = torch.cat([enc_output, cls_token_expended], dim=1)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)
        fc_output = self.fc_out(enc_output)[:, -1, :]

        return fc_output
    

class seq_encoder_only(nn.Module):
    def __init__(self, config):
        super(seq_encoder_only, self).__init__()
        self.d_token = config["d_token"]
        self.num_heads = config["num_heads"]
        self.d_ff = config["d_ff"]
        self.num_layers = config["num_layers"] 
        self.dropout = config["dropout"]
        self.embedding = seq_embedding(self.d_token)
        self.position = PositionalEncoding(self.d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_token))
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_token, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)])
        self.fc_out = nn.Sequential(nn.Linear(self.d_token, 100))
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        # tokenlize
        enc_output = self.embedding(x.int())
        # add the positional encoding
        enc_output = self.position(enc_output)

        # went through different stages of attention
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)
            # print(enc_output)

        # add the linear module
        logits = self.fc_out(enc_output)
        distribution = self.softmax(logits)
        x = torch.argmax(distribution, dim=-1)

        return logits.view(-1, 100), x