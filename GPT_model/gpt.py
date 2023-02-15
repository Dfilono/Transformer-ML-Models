import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many sequences ran in parallel
block_size = 256 # what is the maximum length for predictios
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 # number of embedding dimesions
n_head = 6
n_layer = 6
dropout = 0.2
dataset = 'data.txt' 

# read in input file
with open(dataset, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
unique_chars = len(chars)

# map characters to integers
char_to_int = { c:i for i,c in enumerate(chars)}
int_to_char = { i:c for i,c in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s] # take a string as input and output list of integers
decode = lambda l: ''.join([int_to_char[i] for i in l]) # take a list of ints as input and outputs a string

# econdoe input text
data = torch.tensor(encode(text), dtype=torch.long)

# split data up into training and validations sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def  get_batch(sets):
    # generate a small batch of data of inputs x and targets y
    data = train_data if sets == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    # One head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # compute attention scores
        weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #(B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weight = F.softmax(weight, dim = -1) # (B, T, T)
        weight = self.dropout(weight)

        # perform the weighted aggregation of values
        v = self.value(x) # (B, T, head_size)
        out = weight @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out

class MultiHead(nn.Module):
    # multi head attention

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out  = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    # linear layer followed by a non-linearity

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer block: communication followed by computation

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigram Language Model
class Bigram_Language_Model(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly read off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(unique_chars, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        #self.sa_head = MultiHead(4, n_embed//4) # 4 heads of 8 dimensional self-attention
        #self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, unique_chars)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = token_embed + pos_embed # (B, T, C)
        #x = self.sa_head(x) # apply one head of self-attention (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, unique_chars)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # loss function
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new):
        # idx is the (B, T) array of indices in the current context
        for _ in range(max_new):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # focus on last time step
            logits = logits[:, -1, :] # (B, C)

            # softmax to get probabilites
            probs = F.softmax(logits, dim = -1) # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)

        return idx

model = Bigram_Language_Model()
m = model.to(device)

# create PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # every once in a whiel evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample  a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb ,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new = 500)[0].tolist()))