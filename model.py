"""Small LLM Model Definition.

This module defines the components of the transformer architecture used in the small LLM implementation. It includes:
    - Head: A single attention head.
    - MultiHeadAttention: Multiple attention heads running in parallel.
    - FeedForward: A simple feedforward neural network.
    - Block: A complete transformer block that combines multi-head attention and feedforward layers.
    - SimpleLLM: The overall GPT-style language model that uses the defined blocks to process input text and generate output.
    
Usage:
    from model import SimpleLLM
    model = SimpleLLM()
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import n_embd, block_size, device, n_head, n_layer
from dataset import vocab_size
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

class Head(nn.Module):
    """One head of self-attention.
    
    Args:
        head_size: The dimensionality of the attention head (usually n_embd // n_head).
        
    Returns:
        out: The output of the attention head after processing the input.
    """
    def __init__(self, head_size):
        """Initialize the attention head with linear layers for key, query, and value projections, 
        and a lower triangular mask for causal attention.
        
        Args:
            head_size: The dimensionality of the attention head (usually n_embd // n_head).
        
        Returns:
            None. Initializes the layers and mask for the attention head.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """Compute the output of the attention head given the input tensor x.
        
        Args:
            x: A tensor of shape (B, T, n_embd) representing the input embeddings for a batch of sequences.
        
        Returns:
            out: A tensor of shape (B, T, head_size) representing the output of the attention head after 
            processing the input.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""
    def __init__(self, num_heads, head_size):
        """Initialize the multi-head attention module with the specified number of heads and head size.

        Args:
            num_heads: The number of attention heads.
            head_size: The dimensionality of each attention head.

        Returns:
            None. Initializes the attention heads and a linear projection layer for the output.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        """Compute the output of the multi-head attention module given the input tensor x.
        
        Args:
            x: A tensor of shape (B, T, n_embd) representing the input embeddings for a batch of sequences.
        
        Returns:
            out: A tensor of shape (B, T, n_embd) representing the output of the multi-head attention module after processing the input.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity.
    
    Args:
        n_embd: The dimensionality of the input and output embeddings for the feedforward network.
    
    Returns:
        out: The output of the feedforward network after processing the input.
    """
    def __init__(self, n_embd):
        """Initialize the feedforward network with the specified embedding dimensionality.

        Args:
            n_embd: The dimensionality of the input and output embeddings for the feedforward network.

        Returns:
            None. Initializes the layers for the feedforward network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd) # Projection back to embedding dim
        )

    def forward(self, x):
        """Compute the output of the feedforward network given the input tensor x.
        
        Args:
            x: A tensor of shape (B, T, n_embd) representing the input embeddings for a batch of sequences.
        
        Returns:
            out: A tensor of shape (B, T, n_embd) representing the output of the feedforward network after processing the input.
        """
        return self.net(x)

class Block(nn.Module):
    """A complete Transformer Block.
    
    Args:
        n_embd: The dimensionality of the input and output embeddings for the block.
        n_head: The number of attention heads in the multi-head attention module.
        
    Returns:
        out: The output of the Transformer block after processing the input.
    """
    def __init__(self, n_embd, n_head):
        """Initialize the Transformer block with the specified embedding dimensionality and number of attention heads.

        Args:
            n_embd: The dimensionality of the input and output embeddings for the block.
            n_head: The number of attention heads in the multi-head attention module.

        Returns:
            None. Initializes the layers for the Transformer block.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Compute the output of the Transformer block given the input tensor x.
        
        Args:
            x: A tensor of shape (B, T, n_embd) representing the input embeddings for a batch of sequences.
        
        Returns:
            out: A tensor of shape (B, T, n_embd) representing the output of the Transformer block after 
            processing the input.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleLLM(nn.Module):
    """The final GPT-style Language Model.
    
    Args:  
        None. The model is initialized with predefined configurations for the embedding size, number of 
        layers, and attention heads.
        
    Returns:
        out: The output of the language model after processing the input text.
    """
    def __init__(self):
        """Initialize the SimpleLLM model with the necessary layers and configurations.
        
        Args:
            None. The model is initialized with predefined configurations for the embedding size, number of 
            layers, and attention heads.

        Returns:
            None. Initializes the layers for the SimpleLLM model.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Compute the output of the SimpleLLM model given the input tensor idx and optional target tensor.
        
        Args:
            idx: A tensor of shape (B, T) containing the input token indices for a batch of sequences.
            targets: An optional tensor of shape (B, T) containing the target token indices for computing the loss.

        Returns:
            logits: A tensor of shape (B, T, vocab_size) representing the raw output scores for each token in the vocabulary.
            loss: A scalar tensor representing the cross-entropy loss between the predicted logits and the target token indices, or None if targets is not provided.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb 
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

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
    ):
        """Generate new text tokens given a starting context and a specified number of new tokens to generate.
        
        Args:
            idx: A tensor of shape (B, T) containing the input token indices for a batch of sequences.
            max_new_tokens: An integer specifying the maximum number of new tokens to generate.
            temperature: Scales the logits before sampling.
            top_k: Restrict sampling to the top-k logits when provided.
            top_p: Restrict sampling to the smallest set of tokens whose cumulative probability exceeds top-p.
            repetition_penalty: Penalize tokens that have already appeared in the running context.

        Returns:
            idx: A tensor of shape (B, T + max_new_tokens) containing the input token indices concatenated with the generated token indices.
        """
        processors = LogitsProcessorList()
        if temperature and temperature != 1.0:
            processors.append(TemperatureLogitsWarper(float(temperature)))
        if top_k:
            processors.append(TopKLogitsWarper(int(top_k)))
        if top_p and top_p < 1.0:
            processors.append(TopPLogitsWarper(float(top_p)))
        if repetition_penalty and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(float(repetition_penalty)))

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if processors:
                logits = processors(idx_cond, logits)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
