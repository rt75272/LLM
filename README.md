# LLM — Large Language Model from Scratch

A simple character-level GPT-style decoder-only language model built from scratch using **PyTorch**. This project provides a minimalistic, modular implementation of a Transformer network, intended for educational purposes and experimentation.


## What is an LLM?

Large Language Models (LLMs) are artificial intelligence systems designed to understand, generate, and interact with human language. At their core, they are highly complex deep learning models trained on vast amounts of text to recognize patterns in language.

### How are they built?

Building a modern GPT-style LLM broadly follows these steps:

1. **Tokenization:** Raw text is broken down into small manageable pieces called "tokens" (which can be characters, sub-words, or whole words). Each token is mapped to a numeric ID.
2. **Embedding:** Those integer IDs are converted into dense mathematical vectors (embeddings) so the neural network can process their meanings.
3. **The Transformer Architecture:** The model uses **self-attention mechanisms** to evaluate how important every token in a sequence is to every other token. This allows the model to deeply understand grammatical structure and context over long distances.
4. **Pre-Training (Next-Token Prediction):** The core training objective is deceptively simple: given a sequence of tokens, predict the very *next* token. We calculate a loss (how wrong the prediction was) and use backpropagation and optimizers (like AdamW) to adjust the model's millions/billions of internal weights.
5. **Inference (Autoregressive Generation):** Once training is complete, we feed a starting prompt into the model. It predicts the most likely next token, appends it to the context window, and repeats the process over and over to generate coherent text.


## Architecture

The model implements a standard decoder-only Transformer with the following components:
- **Token Embedding:** Translates raw character integers into continuous vectors.
- **Positional Embedding:** Injects order information into the sequence.
- **Transformer Blocks:** A sequence of blocks containing:
  - **Multi-Head Self-Attention:** Specifically Causal Self-Attention (using a mask to prevent looking into the future).
  - **FeedForward Network:** A linear layer followed by ReLU activation and another linear projection.
  - **LayerNorm & Residual Connections:** Pre-norm architecture for stable training.
- **LM Head:** A final linear mapping to predict the next character over the vocabulary.



## Project Layout

The codebase has been refactored into modular components for clarity and reusability:

```text
config.py    - Hyperparameters and device setup (GPU/CPU)
dataset.py   - Tiny character-level dataset and tokenizer, along with batching logic
model.py     - PyTorch modules for Head, MultiHeadAttention, FeedForward, Block, and the SimpleLLM
train.py     - Training loop with AdamW optimizer and loss evaluation
generate.py  - Interactive chat loop for pretrained and local model backends
main.py      - Entry point that launches chat or trains the local toy model
```


## Quick Start

Make sure you have PyTorch installed (e.g., via `pip install torch`).

Run the main script to launch the chatbot:

```bash
python main.py
```

### Expected Flow
1. **Setup:** Detects available device (e.g., `cuda` or `cpu`).
2. **Chat:** By default, loads a pretrained conversational model and starts an interactive terminal chat session.
3. **Fallback:** If you switch `chat_backend` to `'local'`, the project trains the scratch GPT-style model and then launches chat.

## Conversation Improvements

The chatbot now uses a few inference-time libraries and controls to make responses smoother:

- **Pretrained conversation model:** Uses `HuggingFaceTB/SmolLM2-360M-Instruct` by default for actual chat quality.
- **Code-capable model routing:** Sends coding requests to `Qwen/Qwen2.5-Coder-0.5B-Instruct` so Python and other code prompts use a model tuned for code generation.
- **Transformers logits processors:** Applies temperature scaling, top-k sampling, top-p nucleus sampling, and repetition penalties during decoding.
- **Datasets dialogue loading:** Can pull in conversational examples from `daily_dialog` when enabled in `config.py`, with a built-in fallback corpus used by default.
- **Conversation history:** Recent turns are folded back into the prompt so responses stay more consistent across multiple exchanges.
- **Rich terminal UI:** Displays a cleaner interactive chat interface in the terminal.


## Configuration

You can tweak the hyperparameters in `config.py` to change the size and performance of the model:

- `batch_size`: Number of independent sequences processed in parallel (currently 64).
- `block_size`: Maximum context length (currently 256).
- `max_iters`: Total training iterations.
- `learning_rate`: Default set to 3e-4.
- `n_embd`: Embedding dimension (currently 384).
- `n_head`: Number of self-attention heads (currently 6).
- `n_layer`: Number of Transformer blocks (currently 6).

If a GPU is available, the configuration automatically utilizes `cuda` for accelerated training.

For the best chatbot behavior, keep `chat_backend = 'pretrained'` in `config.py`. The local backend is still useful as a learning/demo path, but it is not competitive as a real chatbot.
