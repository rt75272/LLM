"""Small LLM Text Generation Script.

This script provides functionality to generate text using a trained SimpleLLM model.

Usage:
    from generate import generate_text
    generate_text(model)
"""
import torch
from config import device
from dataset import decode

def generate_text(model):
    """Generate text using the trained model.
    
    Args:
        model: The trained SimpleLLM model.
    
    Returns:
        None. Prints the generated text to the console.
    """
    print("\n--- Generating Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=200)[0].tolist()
    print(decode(generated_tokens))
