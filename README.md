# LLM — Large Language Model from Scratch

A GPT-style decoder-only transformer built entirely in **pure Python** with
**no third-party packages**.  Everything — matrix algebra, back-propagation,
the Adam optimiser, and autoregressive text generation — is implemented from
first principles using only Python's standard library (`math`, `random`,
`json`, `struct`).

---

## Architecture

```
Token Embedding  +  Positional Embedding
        │
 ┌──────▼──────┐  ×  N
 │  LayerNorm  │
 │  CausalSelf │   multi-head masked self-attention
 │  Attention  │
 │  + residual │
 ├─────────────┤
 │  LayerNorm  │
 │  FeedForward│   Linear → GELU → Linear  (4× expansion)
 │  + residual │
 └──────┬──────┘
        │
   LayerNorm  →  Linear (LM head)  →  Cross-Entropy Loss
```

Trained with **Adam** and **backpropagation** implemented manually through
every layer.

---

## Project layout

```
llm/
  __init__.py      public API
  matrix.py        pure-Python matrix / vector operations
  tokenizer.py     character-level tokenizer (fit / encode / decode / save / load)
  layers.py        Embedding, Linear, LayerNorm, CausalSelfAttention,
                   FeedForward, TransformerBlock  (forward + backward)
  model.py         GPT model + cross_entropy_loss
  optimizer.py     Adam optimiser
  trainer.py       training loop  (get_batch, train)
  generate.py      autoregressive generation  (temperature + top-k sampling)
main.py            end-to-end demo (train on Shakespeare excerpt, then generate)
requirements.txt   empty — no dependencies
```

---

## Quick start

```bash
python main.py
```

Expected output (truncated):

```
============================================================
  From-Scratch LLM  (pure Python, no third-party packages)
============================================================
  Corpus  : 1,406 characters
  Vocab   : 43 unique characters
  Tokens  : 1,406
  Params  : 29,248
============================================================

Training for 300 steps …

  step    100/300  loss 3.0434
  step    200/300  loss 2.5681
  step    300/300  loss 2.4616

Final loss : 2.4616

============================================================
  Generating 300 characters  (prompt: 'To be')
============================================================

To be ...
```

---

## Using the library

```python
from llm import CharTokenizer, GPT, train, generate

# 1. Tokenise your corpus
tokenizer = CharTokenizer().fit(my_text)
data = tokenizer.encode(my_text)

# 2. Build a model
model = GPT(
    vocab_size  = tokenizer.vocab_size,
    n_embd      = 64,   # embedding dimension
    n_head      = 4,    # attention heads
    n_layer     = 4,    # transformer blocks
    block_size  = 64,   # context window
)

# 3. Train
losses = train(model, data, n_steps=1000, lr=3e-3)

# 4. Generate
tokens = generate(model, tokenizer.encode("Once upon"), max_new_tokens=200,
                  temperature=0.8, top_k=10)
print(tokenizer.decode(tokens))

# 5. Save / load
model.save("model.bin")
tokenizer.save("tokenizer.json")

model2    = GPT.load("model.bin")
tokenizer2 = CharTokenizer.load("tokenizer.json")
```

---

## Implementation notes

| Component | Details |
|-----------|---------|
| **Tokenizer** | Character-level; vocabulary built from unique characters in the corpus |
| **Embeddings** | Token + learned positional embeddings |
| **Attention** | Multi-head causal (masked) self-attention; backward derived analytically |
| **Normalisation** | Pre-norm LayerNorm with analytic backward pass |
| **Activation** | GELU with exact gradient |
| **Optimiser** | Adam (β₁=0.9, β₂=0.999) with optional weight decay |
| **Generation** | Autoregressive with temperature scaling and top-k filtering |
| **Persistence** | Binary format via `struct` (model) and JSON (tokenizer) |
