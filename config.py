"""Configuration file for the SimpleLLM model. 

This file contains all the hyperparameters and setup configurations needed to train the model.

Usage:
    from config import batch_size, block_size, learning_rate, device
"""
import torch

batch_size = 64      # How many text chunks to process in parallel
block_size = 256     # Maximum context length (how many characters it looks back)
max_iters = 5000     # Training iterations
learning_rate = 3e-4 # Lower learning rate
eval_iters = 200     # How often to report loss
n_embd = 384         # Embedding dimension (size of the vector representing a token)
n_head = 6           # Number of attention heads
n_layer = 6          # Number of transformer blocks

# Use GPU if available, otherwise fallback to CPU
device = ''
if torch.cuda.is_available():
     device = 'cuda' 
else:
     device = 'cpu'
