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
from config import batch_size, block_size, device, use_hf_dialogue_dataset

BASE_TEXT = """
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
"""

FALLBACK_DIALOGUES = [
    ("Hello", "Hi. How can I help you today?"),
    ("What is a transformer model?", "A transformer model uses attention to relate tokens across a sequence and predict the next token more effectively."),
    ("Can you explain machine learning simply?", "Machine learning is a way for computers to learn patterns from examples instead of following only hand-written rules."),
    ("Why is Python popular?", "Python is popular because it is readable, productive, and has strong libraries for data science and automation."),
    ("How do I train a language model?", "You collect text, tokenize it, train on next-token prediction, and tune the optimizer and context length carefully."),
    ("What does overfitting mean?", "Overfitting means the model memorizes the training data too closely and performs poorly on new examples."),
    ("Can you be more concise?", "Yes. I can keep answers shorter and more direct."),
    ("How does attention work?", "Attention scores each token against the others so the model can focus on the most relevant context."),
]


def _load_dialogue_text():
    fallback_text = "\n".join(
        f"User: {user}\nAssistant: {assistant}" for user, assistant in FALLBACK_DIALOGUES
    )
    if not use_hf_dialogue_dataset:
        return fallback_text

    try:
        from datasets import load_dataset
    except ImportError:
        return fallback_text

    try:
        dataset = load_dataset("daily_dialog", split="train[:1%]")
    except Exception:
        return fallback_text

    dialogue_samples = []
    for sample in dataset:
        turns = sample.get("dialog") or []
        paired_turns = []
        for index in range(0, len(turns) - 1, 2):
            user = turns[index].strip()
            assistant = turns[index + 1].strip()
            if user and assistant:
                paired_turns.append(f"User: {user}\nAssistant: {assistant}")
        if paired_turns:
            dialogue_samples.append("\n".join(paired_turns))

    if not dialogue_samples:
        return fallback_text

    dialogue_text = "\n".join(dialogue_samples[:300])
    return f"{fallback_text}\n{dialogue_text}"


text = f"{BASE_TEXT}\n{_load_dialogue_text()}\n" * 200

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
