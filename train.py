"""Small LLM Training Script. 

Usage:
    from train import train_model
    model = train_model(model)  # where 'model' is an instance of SimpleLLM
"""
import torch
from config import learning_rate, max_iters, eval_iters
from dataset import get_batch

def train_model(model):
    """Train the SimpleLLM model on the dataset.
    
    Args:
        model: An instance of the SimpleLLM model to be trained.
        
    Returns:
        The trained model after the training loop is complete."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Starting training...")
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            with torch.no_grad():
                xb, yb = get_batch()
                _, loss = model(xb, yb)
                print(f"Step {iter}: Loss {loss.item():.4f}")
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Final Loss: {loss.item():.4f}")
    return model
