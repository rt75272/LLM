"""Small LLM Main Script.

This is a simple implementation of a character-level GPT-style language model 
built from scratch using PyTorch.  

Usage:
    python main.py
"""
import torch
from config import device
from model import SimpleLLM
from train import train_model
from generate import generate_text

def main():
    print(f"Using device: {device}")
    model = SimpleLLM().to(device) # Initialize model and move it to GPU.
    model = train_model(model) # Train the model.
    generate_text(model) # Generate text using the trained model.

# The big red activation button.
if __name__ == '__main__':
    main()
