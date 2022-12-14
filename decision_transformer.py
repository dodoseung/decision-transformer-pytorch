import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Decision Transformer
class DecisionTransformer(nn.Module):
    def __init__(self, num_decoder_layer=12, num_heads=12, d_ff=3072, dropout=0.1, 
                 max_seq_len=1024, h_dim=768, state_dim=128, action_dim=128, device="cpu"):
        super(DecisionTransformer, self).__init__()
        # Device
        self.device = device

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Input embedding layers
        self.rtg_embedding = nn.Linear(1, h_dim)
        self.state_embedding = nn.Linear(state_dim, h_dim)
        self.action_embedding = nn.Linear(action_dim, h_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, h_dim)

        # Decoder
        self.decoder = Decoder(num_decoder_layer, h_dim, num_heads, d_ff, dropout)
        
        # Output prediction layers
        self.rtg_prediction = nn.Linear(h_dim, 1)
        self.state_prediction = nn.Linear(h_dim, state_dim)
        self.action_prediction = nn.Linear(h_dim, action_dim)
    
    def forward(self, trg):
        # Token embedding
        pos = torch.arange(0, trg.size(-1)).unsqueeze(0)
        # trg_emb = self.token_embedding(trg) + self.positional_embedding(pos)
        # trg_emb = self.dropout(trg_emb)

        # Decoder
        # decoder_out = self.decoder(trg_emb)
        
        # Transform to character
        # out = self.out_layer(decoder_out)
        
        # return out

# Decoder
class Decoder(nn.Module):
    def __init__(self, num_layer, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layer)]) 

    def forward(self, trg):
        # Target mask
        trg_mask = self.look_ahead_mask(trg)

        # Encoder layers
        for layer in self.layers:
            trg = layer(trg, trg_mask)
            
        return trg

    # Set the look ahead mask
    # seq: (batch, seq_len)
    # mask: (batch, 1, seq_len, seq_len)
    # Pad -> True
    def look_ahead_mask(self, seq):
        # Set the look ahead mask
        # (batch, seq_len, seq_len)
        seq_len = seq.shape[1]
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask)
        mask = mask.bool().to(self.device)

        return mask

# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.dropout = dropout

    def forward(self, trg, trg_mask):
        # Masked multi head attention
        out = self.layer_norm1(trg)
        out = self.masked_multi_head_attention(out, out, out, trg_mask)
        residual = out
        
        # Position wise feed foward
        out = self.layer_norm2(out)
        out = self.position_wise_feed_forward(out, self.dropout)
        out = residual + out

        return out

# Masked Multi head attention
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
        
        # Define w_q, w_k, w_v, w_o
        self.weight_q = nn.Linear(self.d_model, self.d_model)
        self.weight_k = nn.Linear(self.d_model, self.d_model)
        self.weight_v = nn.Linear(self.d_model, self.d_model)
        self.weight_o = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        # Batch size
        batch_size = query.shape[0]
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.weight_q(query)
        key = self.weight_k(key)
        value = self.weight_v(value)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k)
        key = key.view(batch_size, -1, self.num_heads, self.d_k)
        value = value.view(batch_size, -1, self.num_heads, self.d_k)
        
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)
        
        # Get the scaled attention
        # (batch, h, query_len, d_k) -> (batch, query_len, h, d_k)
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = torch.transpose(scaled_attention, 1, 2).contiguous()

        # Concat the splitted attentions
        # (batch, query_len, h, d_k) -> (batch, query_len, d_model)
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        # Get the multi head attention
        # (batch, query_len, d_model) -> (batch, query_len, d_model)
        multihead_attention = self.weight_o(concat_attention)
        
        return multihead_attention
    
    # Query, key, and value size: (batch, num_heads, seq_len, d_k)
    # Mask size(optional): (batch, 1, seq_len, seq_len)   
    def scaled_dot_product_attention(self, query, key, value, mask):
        # Get the q matmul k_t
        # (batch, h, query_len, d_k) dot (batch, h, d_k, key_len)
        # -> (batch, h, query_len, key_len)
        attention_score = torch.matmul(query, torch.transpose(key, -2, -1))

        # Get the attention score
        d_k = query.size(-1)
        attention_score = attention_score / math.sqrt(d_k)

        # Get the attention wights
        attention_score = attention_score.masked_fill(mask==0, -1e10) if mask is not None else attention_score
        attention_weights = F.softmax(attention_score, dim=-1, dtype=torch.float)

        # Get the attention value
        # (batch, h, query_len, key_len) -> (batch, h, query_len, d_k)
        attention_value = torch.matmul(attention_weights, value)
        
        return attention_value

# Position wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return out
