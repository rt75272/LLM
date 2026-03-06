"""Small LLM Main Script.

This is a simple implementation of a character-level GPT-style language model 
built from scratch using PyTorch.  

Usage:
    python main.py
"""
import torch
from config import chat_backend, device
from model import SimpleLLM
from train import train_model
from generate import chat

def main():
    print(f"Using device: {device}")
    if chat_backend == 'pretrained':
        chat()
        return

    model = SimpleLLM().to(device) # Initialize model and move it to GPU.
    model = train_model(model) # Train the model.
    chat(model) # Start an interactive chat session.

# The big red activation button.
if __name__ == '__main__':
    main()
