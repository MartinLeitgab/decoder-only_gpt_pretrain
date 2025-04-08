import torch
import torch.nn as nn
import torch.nn.functional as F

# set the random seed, for reproducibility
torch.manual_seed(42)

# device: where to execute computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# windows can access GPU


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


#----------
# load input dataset

with open("pretrain_data/input.txt") as f:
        text = f.read()
#print(text)
        
        
#----------
# load input dataset using OpenAI tiktoken encoder

import tiktoken
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

