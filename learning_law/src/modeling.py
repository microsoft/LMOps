import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super(RMSLayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        x = x / (rms + self.eps)
        return self.weight * x + self.bias


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_head = config["num_head"]
        self.head_dim = self.hidden_size // self.num_head
        
        self.register_buffer("casual_mask", 
            torch.tril(torch.ones(1, 1, config["max_len"], config["max_len"], dtype=torch.long)), persistent=False)
        
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def split_heads(self, x):
        x = x.view(x.size(0), x.size(1), self.num_head, self.head_dim)
        x = x.transpose(1, 2)
        return x # [batch_size, num_head, seq_len, head_dim]
    
    def merge_heads(self, x):
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), self.hidden_size)
        return x

    def forward(self, hidden_states):
        # Self-Attention
        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn = torch.masked_fill(attn, self.casual_mask == 0, torch.finfo(torch.float32).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn, v) # [batch_size, num_head, seq_len, head_dim]
        
        attn_output = self.merge_heads(attn_output)
        
        hidden_states = self.w_o(attn_output)
        
        return hidden_states


class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.activation = F.relu
        
        self.mlp_1 = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=False)
        self.mlp_2 = nn.Linear(4 * self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        x = hidden_states
        x = self.mlp_1(x)
        x = self.activation(x)
        x = self.mlp_2(x)
        return x
    
    
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config["hidden_size"]
        
        self.attn = Attention(config)
        self.ffn = FFN(config)
        
        self.ln_1 = RMSLayerNorm(self.hidden_size)
        self.ln_2 = RMSLayerNorm(self.hidden_size)
    
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    

class BaseTransformer(nn.Module):
    def __init__(self, config):
        super(BaseTransformer, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.max_len = config["max_len"]
        self.num_layers = config["num_layers"]
        self.word_embed = nn.Embedding(config["vocab_size"], self.hidden_size)
        self.pos_embed = nn.Embedding(config["max_len"], self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, config["vocab_size"], bias=False)
        self.ln_end = RMSLayerNorm(self.hidden_size)
        
        self.blocks = nn.ModuleList([Block(config) for i in range(self.num_layers)])
        
        # self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0)
                if module.padding_idx is not None:
                    nn.init.constant_(module.weight[module.padding_idx], 0.0)
            else:
                pass
        
    def forward(self, input_ids):
        input_embed = self.word_embed(input_ids)
        pos_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        pos_embed = self.pos_embed(pos_ids)
        hidden_states = input_embed + pos_embed
        
        for l, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
        
        hidden_states = self.ln_end(hidden_states)
        
        output = self.lm_head(hidden_states)
        
        return output
