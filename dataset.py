""" Dataset and Tokenizer for Character-Level Language Model.

This code defines a simple dataset and tokenizer for a character-level language model. 
It includes a sample text corpus, creates a vocabulary of unique characters, and provides 
functions to encode and decode text. The `get_batch` function generates random batches of 
input and target sequences for training the model.

Usage:
    from dataset import get_batch, decode
    xb, yb = get_batch()
    print(decode(xb[0].tolist()), "->", decode(yb[0].tolist()))
"""
import torch
from config import block_size, batch_size, device

text = """
The quick brown fox jumps over the lazy dog. 
To be, or not to be, that is the question.
Artificial intelligence is the simulation of human intelligence.
Machine learning allows computers to learn without explicit programming.
A neural network is a network or circuit of biological neurons, or, in a modern sense, an artificial neural network, composed of artificial neurons or nodes.
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.
The history of Natural Language Processing generally started in the 1950s, although work can be found from earlier periods.
Transformers have revolutionized the field of NLP by enabling the processing of sequences in parallel.
Attention is all you need: the architectural breakthrough that made modern large language models possible.
""" * 500 # Duplicating it to simulate a larger corpus

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    """Generate a batch of input and target sequences for training.
    
    Args:
        None.   
    
    Returns:    
        x: A tensor of shape (batch_size, block_size) containing input sequences.
        y: A tensor of shape (batch_size, block_size) containing target sequences (input shifted by one character).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
