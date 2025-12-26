import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # the maximum context length (Time)
max_iters = 5000 # total training steps (gradient updates)
eval_interval = 300 # how often loss is evaluated
learning_rate = 1e-3 # learning rate (step size of gradient descent)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available
eval_iters = 200 # how many batches to average over when checking loss
n_embed = 32 # embedding dimension

torch.manual_seed(42)

with open('input.txt' , 'r' , encoding='utf-8') as f:
    text = f.read()

# unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { c:i for i , c in enumerate(chars) }
itos = { i:c for i , c in enumerate(chars) }
# encode and decode the text
encode = lambda s: [stoi[c] for c in s]  # encoder (string -> list of ints)
decode = lambda l: ''.join([itos[i] for i in l])  # decoder (list of ints -> string)

# train and test splits
data = torch.tensor(encode(text) , dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x , y = x.to(device) , y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X , Y = get_batch(split)
            _ , loss = model(X , Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# head
class Head(nn.Module):
    '''one head of self-attention'''
    def __init__(self , head_size):
        super().__init__()
        self.key = nn.Linear(n_embed , head_size , bias=False)
        self.query = nn.Linear(n_embed , head_size , bias=False)
        self.value = nn.Linear(n_embed , head_size , bias=False)
        self.register_buffer('tril' , torch.tril(torch.ones(block_size,block_size)))

    def forward(self , x):
        B,T,C = x.shape
        k , q = self.key(x) , self.query(x)  # (B,T,C)
        # computing the affinities (attention scores)
        weights = q @ k.transpose(-2,-1) * C**-0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))  # (B,T,T)  # decoder block
        weights = F.softmax(weights , dim=-1)  # (B,T,T)
        # weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = weights @ v  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self , num_heads , head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self , x):
        return torch.cat([h(x) for h in self.heads] , dim=-1)

class FeedForward(nn.Module):
    ''' simple linear layer followed by a non-linearity '''
    def __init__(self , n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed , 4 * n_embed) ,
            nn.ReLU() ,
        )

    def forward(self , x):
        return self.net(x)

# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size , n_embed)
        self.position_embedding_table = nn.Embedding(block_size , n_embed)
        self.sa_heads = MultiHeadAttention(4 , n_embed//4)  # 4 heads of 8-dimensional self-attention
        self.feed_forward = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed , vocab_size)
    
    def forward(self , index , targets=None):
        B , T = index.shape
        # index & targets are (B,T) tensor of integers
        tok_embed = self.token_embedding_table(index)  # (B,T,C)
        pos_embed = self.position_embedding_table(torch.arange(T , device=device))  # (T,C)
        x = tok_embed + pos_embed  # (B,T,C)
        x = self.sa_head(x)  # applying one head of self-attention
        x = self.feed_forward(x)  # applying the feed-forward network
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B , T , C = logits.shape
            logits , targets = logits.view(B*T , C) , targets.view(B*T)
            loss = F.cross_entropy(logits , targets)

        return logits , loss

    def generate(self , index , max_new_tokens):
        # index is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get predictions
            logits , loss = self(index_cond)
            # focus only on the last time step
            logits = logits[: , -1 , :]  # (B,C)
            # applying softmax
            probs = F.softmax(logits , dim=-1) # (B,C)
            # sample from the distribution
            index_next = torch.multinomial(probs , num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            index = torch.cat((index,index_next) , dim=1) # (B,T+1)
        return index

model = BigramLanguageModel().to(device)

optimizer = torch.optim.AdamW(model.parameters() , lr=learning_rate)
for iter in range(max_iters):
    
    # evaluate the loss on train and val sets every eval_interval steps
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample batch of data
    xb , yb = get_batch('train')

    # evaluate the loss
    logits , loss = model(xb , yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1) , dtype=torch.long , device=device)
print(decode(model.generate(context , max_new_tokens=500)[0].tolist()))
