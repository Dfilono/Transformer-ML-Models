import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many sequences ran in parallel
block_size = 8 # what is the maximum length for predictios
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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


# Bigram Language Model
class Bigram_Language_Model(nn.Module):

    def __init__(self, unique_chars):
        super().__init__()
        # each token directly read off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(unique_chars, unique_chars)

    def forward(self, idx, targets = None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

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
            # get predictions
            logits, loss = self(idx)

            # focus on last time step
            logits = logits[:, -1, :] # (B, C)

            # softmax to get probabilites
            probs = F.softmax(logits, dim = -1) # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)

        return idx

model = Bigram_Language_Model(unique_chars)
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