import torch
import mmap
import random
import pickle
import argparse
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

block_size = 64
batch_size = 128
max_iters = 2000
learning_rate = 3e-4
eval_iters = 200
n_embd = 384 # Stores the vectors of the token
n_head = 8
n_layer = 8 # Number of decoder blocks
dropout = 0.2

chars = ""
with open('training/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)
vocab_size

string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l)

class Head(nn.Module):
    """Single head of multi-head"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # input size (batch, time-step, channels)
        # output size (batch, time-step, head-size)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # We are calculating Scaled Score = (Query x Key(Transpose)) x 1/√d, where 'd' is the dimension of the Q and K vectors.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # We mask the future results so that the model not look into the future, using upper triangle inf matrix
        wei = wei.softmax(dim=-1) # Applying softmax to get the probability distribution
        wei = self.dropout(wei) # Avoid overfitting
        # perform weighted aggregation of the values
        v = self.value(x) # The values
        out = wei @ v # Matmul on Attention Weight and Values
        return out # Finally we get the output
    

class MultiHeadAttention(nn.Module):
    """Multiple heads of Self-Attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Each head computing in parallel
        self.proj = nn.Linear(head_size * num_heads, n_embd) # After each head completes process, combine those into single vector
        self.dropout = nn.Dropout(dropout) # Avoid overfitting
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # ReLu = max(0, x)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout) # drop few neurons for better learning and avoid overfitting
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Block: Communication follower by computation"""
    
    def __init__(self, n_embd, n_head):
        # n_embd: Embedding Dimensions, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Layer Normalization
        self.ln2 = nn.LayerNorm(n_embd) # Layer Normalization
    
    # Self-Attention -> Normalization -> FeedForward -> Normalization
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

# GPT Architechture, Just a little scaled down version of Transformer Architechture
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
    
    # Weight Initialization using S.D, Goal: Avoid Gradient Explosion or Vanish
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # Forward Pass
    def forward(self, index, targets=None):
        B, T = index.shape

        # index and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

model = GPTLanguageModel(vocab_size)

print("Loading model parameters...")
with open('model/model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model parameters loaded.")

m = model.to(device)

while True:
    prompt = input("Enter your prompt: ")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=200)[0].tolist())
    print(f"Generated text:\n{generated_chars}")