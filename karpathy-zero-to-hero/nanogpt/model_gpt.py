import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self , n_dim , bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self , x):
        return F.layer_norm(input=x , normalized_shape=self.weight.shape , weight=self.weight , bias=self.bias , eps=1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self , n_embed , n_heads , block_size , dropout):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embed , 3 * n_embed , bias=False)
        # output projection
        self.proj = nn.Linear(n_embed , n_embed , bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout = dropout
        self.register_buffer("tril" , torch.tril(torch.ones(block_size , block_size)))

    def forward(self , x):
        B,T,C = x.shape

        q,k,v = self.attn(x).split(self.n_embed , dim=2)
        k = k.view(B , T , self.n_heads , C // self.n_heads).transpose(1,2) # (B,Nh,T,hs)
        q = q.view(B , T , self.n_heads , C // self.n_heads).transpose(1,2) # (B,Nh,T,hs)
        v = v.view(B , T , self.n_heads , C // self.n_heads).transpose(1,2) # (B,Nh,T,hs)

        # Self-Attention
        attention_scores = (q @ k.transpose(-2,-1)) * C**-0.5
        attention_scores = attention_scores.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))
        attention_scores = F.softmax(attention_scores , dim=-1)
        attention_scores = self.attn_dropout(attention_scores)
        y = attention_scores @ v # (B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-concatenate heads

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self , n_embed , dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embed , 4 * n_embed , bias=False)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * n_embed , n_embed , bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self , n_embed , n_heads , block_size , dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embed , bias=False)
        self.attention = CausalSelfAttention(n_embed , n_heads , block_size , dropout)
        self.ln2 = LayerNorm(n_embed , bias=False)
        self.mlp = MLP(n_embed , dropout)

    def forward(self , x):
        x += self.attention(self.ln1(x))
        x += self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.1
    bias: bool = False

class GPT(nn.Module):
    def __init__(self , config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size , config.n_embed),
            wpe = nn.Embedding(config.block_size , config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config.n_embed , config.n_heads , config.block_size , config.dropout) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embed , bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embed , config.vocab_size , bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # initializing all weights
        self.apply(self._init_weights)

    def _init_weights(self , module):
        if isinstance(module , nn.Linear):
            torch.nn.init.normal_(module.weight , mean=0.0 , std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module , nn.Embedding):
            torch.nn.init.normal_(module.weight , mean=0.0 , std=0.02)

    def forward(self , idx):
        device = idx.device
        B,T = idx.shape

        position = torch.arange(0 , T , dtype=torch.long , device=device)

        # forward the GPT model
        tok_emb , pos_emb = self.transformer.wte(idx) , self.transformer.wpe(position)  # (B,T,C) , (T,C)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1,logits.size(-1)) , idx.view(-1))

        return logits , loss

    @torch.no_grad()
    def generate(self , idx , max_new_tokens , temperature=1.0 , top_k=None):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits , _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v , _ = torch.topk(logits[0, -1, :] , top_k)
                logits[logits < v[:,[-1]]] = -float('inf')
            
            # applying softmax to get probabilities
            probs = F.softmax(logits , dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs , num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx , idx_next) , dim=1)

        return idx
