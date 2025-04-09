import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken



# set the random seed, for reproducibility
torch.manual_seed(42)

# device: where to execute computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# windows can access GPU

def get_batch(source): # dependency on device
        """ get batch of size block_size from source """

        # generate `batch_size` random offsets on the data
        ix = torch.randint(len(source)-block_size, (batch_size,) )
        # collect `batch_size` subsequences of length `block_size` from source, as data and target
        x = torch.stack([torch.tensor(source[i:i+block_size]) for i in ix]) # adjusted from tutorial
        # target is just x shifted right (ie the predicted token is the next in the sequence)
        y = torch.stack([torch.tensor(source[i+1:i+1+block_size]) for i in ix]) # adjusted from tutorial
        return x.to(device), y.to(device)


                
#---------
# hyper parameters
# how often to do an evaluation step
eval_interval = 100

# number of training iterations
max_iters = 500

# optimizer's learning rate
learning_rate=1e-4

# minibatch size, how many inputs to 'pack' per iteration 
batch_size = 3

head_size=4


#---------
# GPT specific parameters

# block size is the maximum sequence length used as input.
# E.g. for block_size 4 and input ABCD, we have training samples A->B, AB->C, ABC->C, ABCD->E
block_size = 4

# size of the embeddings
n_embd = 16

# number of attention heads in Multi-Attention mechanism (the Nx in the GPT decoder diagram)
n_head = 6

# depth of the network as number of decoder blocks.
# Each block contains a normalization, an attention and a feed forward unit
n_layer = 6

# dropout rate (variable p) for dropout units
dropout = 0.2


#==========
# decoder head implementation
class Head(nn.Module):

        def __init__(self, head_size):
                super().__init__()
                self.key   = nn.Linear(n_embd, head_size, bias=False)
                self.query = nn.Linear(n_embd, head_size, bias=False)
                self.value = nn.Linear(n_embd, head_size, bias=False)
                self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
                self.dropout = nn.Dropout(dropout)
                #Note: this dropout randomly prevents some tokens from communicating with each other
                
                def forward(self, x):
                        B,T,C = x.shape
                        k = self.key(x) #shape (B,T, head_size)
                        q = self.query(x) #shape (B,T, head_size)
                        v = self.value(x) #shape (B,T, head_size)
                        
                        #compute self-attention scores
                        wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
                        wei *= C**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1
                        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)
                        wei = F.softmax(wei, dim=-1) # (B, T, T)
                        wei = self.dropout(wei)
                        
                        #perform weighted aggregation of values
                        out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
                        return out

                
# multi-head implementation
class MultiHeadAttention(nn.Module):
        """ Multi-head attention as a collection of heads with concatenated outputs."""
        def __init__(self, num_heads, head_size):
                super().__init__()
                self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
                self.proj  = nn.Linear(head_size*num_heads, n_embd) # combine all head outputs
                self.dropout = nn.Dropout(dropout)
                
        def forward(self, x):
                out = torch.cat([head(x) for head in self.heads], dim=-1)
                out = self.proj(out)
                out = self.dropout(out)
                return out

                
# feed forward network- single layer deep neural network
class FeedForward(nn.Module):
        """ the feed forward network (FFN) in the paper"""
        
        def __init__(self, n_embd):
                super().__init__()
                # Note: in the paper (section 3.3) we have d_{model}=512 and d_{ff}=2048.
                # Therefore the inner layer is 4 times the size of the embedding layer
                self.net = nn.Sequential(
                        nn.Linear(n_embd, n_embd*4),
                        nn.ReLU(),
                        nn.Linear(n_embd*4, n_embd),
                        nn.Dropout(dropout)
                )
                
        def forward(self, x):
                return self.net(x)
        

# GPT block == multi-head attention and feedforward module:
# to avoid high number of sequential blocks (otherwise network too deep/too hard to train) added skip connections to each block
# now common to apply layer normalization in pre-norm formulation (before attention and FNN, instead of after)

class Block(nn.Module):
        """ Transformer block: comunication (attention) followed by computation (FFN) """
        
        def __init__(self, n_embd, n_head):
                # n_embd: embedding dimension
                # n_heads : the number of heads we'd like to use
                super().__init__()
                head_size = n_embd // n_head
                self.sa = MultiHeadAttention(n_head, head_size)
                self.ffwd = FeedForward(n_embd)
                self.ln1 = nn.LayerNorm(n_embd)
                self.ln2 = nn.LayerNorm(n_embd)
                
        def forward(self, x):
                x = x + self.sa(self.ln1(x))
                x = x + self.ffwd(self.ln2(x))
                return x


#----------
# load input dataset

with open("pretrain_data/input.txt") as f:
        text = f.read()
#print(text)
        
        
#----------
# load input dataset using OpenAI tiktoken encoder

enc = tiktoken.encoding_for_model("gpt-4")
print(enc.n_vocab)
#text = "This is an example sentence."
tokens = enc.encode(text) # transform back to text by decode() function
vocab_size = len(tokens)
#print(vocab_size)

# split train-validation data
#data = encode(text)  #use any encoder here, replaced by tiktoken
n = int(0.9*len(tokens))
train_data, valid_data = tokens[:n], tokens[n:]
print(f" {len(train_data)}, {len(valid_data)}")

#-----------
# positional encodings

token_embedding_table = nn.Embedding(vocab_size, n_embd)    # from tokens to embedding
position_embedding_table = nn.Embedding(block_size, n_embd) # from position to embedding

# test get_batch()
#print(type(train_data))
#print(train_data[1:2])
xb, yb = get_batch(train_data)
print("input:\n",xb)
print("target:\n",yb)


#=========
# main model wrapper

class GPTlite(nn.Module):
        
        def __init__(self, vocab_size):
                super().__init__()
                
                # vocabulary embedding and positional embedding
                self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
                self.position_embedding_table = nn.Embedding(block_size, n_embd)
                
                #sequence of attention heads and feed forward layers
                self.blocks = nn.Sequential( *[Block(n_embd, n_head) for _ in range(n_layer)])
                
                #one layer normalization layer after transformer blocks
                #and one before linear layer that outputs the vocabulary
                self.ln = nn.LayerNorm(n_embd)
                self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
                
                
        def forward(self, idx):
                """ call the model with idx and targets (training) or without targets (generation)"""
                
                #idx and targets are both of shape (B,T)
                B, T = idx.shape
                tok_emb = self.token_embedding_table(idx) #shape (B,T,C)
                pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) #shape (T,C)
                x = tok_emb + pos_emb #shape (B,T,C)
                x = self.blocks(x)
                x = self.ln(x)
                logits = self.lm_head(x) #shape (B,T,C)
                logits = torch.swapaxes(logits, 1, 2) #shape (B,C,T) to comply with CrossEntropyLoss
                return logits


# inference (token generation) function
def generate(self, idx, max_new_tokens):
        """ given a context idx, generate max_new_tokens tokens and append them to idx """
        for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:] #we can never have any idx longer than block_size
                logits = self(idx_cond) #call fwd without targets
                logits = logits[:, :, -1] # take last token. from shape (B, C, T) to (B, C)
                #convert logits to probabilities
                probs = F.softmax(logits, dim=-1) # shape (B, C)
                #randomly sample the next tokens, 1 for each of the previous probability distributions
                #(one could take instead the argmax, but that would be deterministic and boring)
                idx_next = torch.multinomial(probs, num_samples=1) # shape (B, 1)
                #append next token ix to the solution sequence so far
                idx = torch.cat([idx, idx_next], dim=-1) # shape (B, T+1)
        return idx  
                                                                                                        

#=========
# train loop

# instantiate model and copy to compute device
m  = GPTlite(vocab_size).to(device)

# initialize optimizer and perform train loop



"""
for b in range(batch_size): #for every batches
        print(f"\n=== batch {b}:")
        for t in range(block_size): #for each sequence in block
                context = xb[b,:t+1]
                target = yb[b,t]
                print(f"for input {context.tolist()} target is {target.tolist()}")
"""

"""
#-------------
# illustration: Multi-headed Masked Attention


#B, T, C, = 4, 8, 2
B, T, C, = batch_size, block_size, n_embd # modified from tutorial
x = torch.randn(B,T,C) #shape (B,T,C)
print(f"\n x shape \n {x.shape}")

#size C for the embedding size of each token, defined by the n_embed on top;
#size B for the batch size defined above by batch_size;
#size T for the time dimension, or input sequence length, defined by block_size


# option: uniform attention matrix
#attention matrix (lower triangular), a mask used to only show previous items to predict next item
#wei = torch.tril(torch.ones((T,T), dtype=torch.float32, device="cuda:0")) # adjusted from tutorial
wei = torch.tril(torch.ones((T,T), dtype=torch.float32)) # adjusted from tutorial
#wei /= wei.sum(dim=1, keepdim=True)
#tril = torch.tril(torch.ones((T,T), device="cuda:0"))
tril = torch.tril(torch.ones((T,T)))
wei = wei.masked_fill(tril==0, float('-inf') ) 
wei = F.softmax(wei, dim=-1)
print(f"\n wei shape \n {wei.shape}")

# dot product of input x by attention wei to see output of attention head for given x
out = wei @ x   # wei has shape (T,T), then python broadcasting turns dot product shape to: (B,T,T) @ (B,T,C) = (B,T,C) ie ([4, 8, 2])
print(out)
#print(f"\n out shape \n {out.shape}")


# option: non-uniform attention matrix

# Wq, Wk, Wv matrices- projections/linear layer
head_size=4
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# compute Attention(Q,K,V)
k = key(x) #shape (B,T, head_size)- self-attention because all come from same input x (could derive K, Q, V from different sources- then 'cross attention'); also no cross-batch attention
q = query(x) #shape (B,T, head_size)
wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) = (B,T,T)
wei *= head_size**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1; creates more sparse/not one-hot vector with diffused values

# compute output of non-uniform attention matrix- solves problem of non-uniform attention weights per query; aggregated by 'value of importance' for each token
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf')) #tokens only "talk" to previous tokens but different use cases can require custom masks
wei = F.softmax(wei, dim=-1) #equivalent to the normalization above (-inf in upper diagonal will be 0)
v = value(x) # shape (B,T, head_size)
out = wei @ v # shape (B,T,T) @ (B,T,C) --> (B,T,C) # adjusted from tutorial- shape (B, T, T ) @ (B, T, head_size) --> (B,T,head_size)

#print(out)
print(f"\n out shape \n {out.shape}")
"""

