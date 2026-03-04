"""
main.py — end-to-end demonstration of the from-scratch LLM.

What this script does
---------------------
1. Trains a small GPT model on a short excerpt of Shakespeare.
2. Prints the training loss every 100 steps.
3. Generates new text from the trained model.

No third-party packages are used — only Python's standard library.
Run with:
    python main.py
"""

import random
from llm import CharTokenizer, GPT, train, generate

# ---------------------------------------------------------------------------
# Training corpus  (a short Shakespeare excerpt)
# ---------------------------------------------------------------------------

TEXT = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.
"""

# ---------------------------------------------------------------------------
# Hyperparameters  (kept small for pure-Python speed)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32   # context window
N_EMBD     = 32   # embedding / model dimension
N_HEAD     = 2    # attention heads  (head_dim = N_EMBD // N_HEAD = 16)
N_LAYER    = 2    # transformer blocks
N_STEPS    = 300  # training steps
LR         = 5e-3 # learning rate

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    # --- Tokenise --------------------------------------------------------
    tokenizer = CharTokenizer().fit(TEXT)
    data = tokenizer.encode(TEXT)

    print("=" * 60)
    print("  From-Scratch LLM  (pure Python, no third-party packages)")
    print("=" * 60)
    print(f"  Corpus  : {len(TEXT):,} characters")
    print(f"  Vocab   : {tokenizer.vocab_size} unique characters")
    print(f"  Tokens  : {len(data):,}")

    # --- Build model -----------------------------------------------------
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
    )
    print(f"  Params  : {model.num_params():,}")
    print("=" * 60)

    # --- Train -----------------------------------------------------------
    print(f"\nTraining for {N_STEPS} steps …\n")
    losses = train(
        model,
        data,
        n_steps=N_STEPS,
        lr=LR,
        eval_interval=100,
        verbose=True,
    )

    print(f"\nFinal loss : {losses[-1]:.4f}")

    # --- Generate --------------------------------------------------------
    prompt = "To be"
    start_tokens = tokenizer.encode(prompt)

    print("\n" + "=" * 60)
    print(f"  Generating 300 characters  (prompt: {prompt!r})")
    print("=" * 60 + "\n")

    tokens = generate(
        model,
        start_tokens,
        max_new_tokens=300,
        temperature=0.8,
        top_k=10,
    )
    print(tokenizer.decode(tokens))
    print()


if __name__ == "__main__":
    main()
