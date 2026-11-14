# Smart Keyboard: next-word prediction with an RNN and permanent personal memory

A mobile-style keyboard assistant built from scratch with **NumPy only** (no deep-learning framework). It predicts the next word as you type, completes half-typed words, and **keeps learning from you permanently**, the way a phone keyboard does.

- **Base model:** a word-level RNN trained on a text corpus (general language knowledge).
- **Personal memory:** a SQLite database of the word pairs *you* type, with recency decay. It survives restarts.
- **Tkinter UI:** a message box with three clickable suggestions. Every send trains the system.

```
typed text ──► tokenizer ──► WordRNN (context from earlier words) ─┐
                                                                   ├─► blend ──► top-3 suggestions
typed text ──────────────► Personal memory (SQLite) ───────────────┘
                                  ▲
         every sent message updates it permanently
```

---

## Features

| Feature | How |
|---|---|
| Next-word prediction | RNN scores every vocabulary word given the words typed so far |
| Word completion | Candidates are restricted to words starting with the typed prefix (`how ar` → `are`) |
| Permanent memory | Pair counts are committed to `memory.db` after every message |
| Personalization | Blend of RNN and your own habits; your habits take over as evidence grows |
| Forgetting | Counts decay with a 30-day half-life, so old habits fade |
| On-device fine-tuning | Optional tiny SGD step on the RNN weights per message |
| Crash-safe saving | Model files are written to a temp file, then atomically renamed |

---

## Project structure

```
smart_keyboard/
├── common.py          # sigmoid, softmax, Adam, clip_grads, gradcheck (shared toolkit)
├── libs.py            # tokenize_sentences, build_vocab, encode, START/UNK markers
├── WordRNN.py         # the RNN: forward, loss, backward (BPTT), next_probs, evaluate
├── memory.py          # PersonalMemory: SQLite word-pair counts with recency decay
├── SmartKeyboard.py   # SmartKeyboard class: train, save/load, learn, suggest
├── keyboard_ui.py     # Tkinter app
├── corpus.txt         # (optional) your training text, UTF-8
└── keyboard_data/     # created automatically
    ├── model.npz      # RNN weights
    ├── vocab.json     # id → word list
    └── memory.db      # your personal word-pair counts
```

---

## Requirements

- Python 3.10+ (developed on 3.13)
- NumPy: `pip install numpy`
- Tkinter (included with the standard Windows and macOS Python installers; on some Linux systems install the `python3-tk` package)
- SQLite 3.24+ (bundled with current Python versions; the code uses an upsert)

No GPU is needed.

---

## Quick start

1. Put all `.py` files in one folder.
2. *(Optional but recommended)* add a `corpus.txt` next to them (see [Training data](#training-data)). Without it, the app trains on a small templated toy corpus, which is enough to test the pipeline but not to make a useful keyboard.
3. Run the UI:

   ```
   python keyboard_ui.py
   ```

4. **First run only:** training happens in the console and prints progress every 200 steps. When it finishes, the window opens. Later runs load the saved model instantly.

To run the keyboard from the command line without a UI, run `python SmartKeyboard.py`. It trains if needed, then prints sample suggestions.

---

## Using the UI

| Action | Result |
|---|---|
| Type in the text box | The three buttons above it show the best next words (or completions of the word you're typing) |
| Click a suggestion button | The half-typed word is replaced by the suggestion plus a space |
| **Tab** | Accept the first suggestion |
| **Enter** or **Send** | Message goes to the history and the keyboard **learns from it** |
| Checkbox "Also fine-tune the RNN weights" | On: personal memory and RNN both learn. Off: only personal memory learns |

After each send, the status bar shows how many word pairs are stored and, with fine-tuning on, the RNN loss on your message **before → after**.

### Test that it learns

1. Type `see you at the ` and note the suggestions.
2. Send `see you at the library` four or five times.
3. Type `see you at the ` again. `library` should now be near the top.
4. Close the app, reopen it, and repeat step 3. It should still be there.

---

## How it works

### 1. Tokenizing

Text is lowercased and split into sentences. Each sentence gets a `<s>` marker so the model learns which words start sentences. Words are letters and apostrophes (`don't` stays one word; curly `’` is normalized to `'`). The vocabulary keeps the 4,000 most frequent training words; everything else becomes `<unk>`.

### 2. The RNN (`WordRNN.py`)

At each word the network mixes the new input with its memory of earlier words:

```
x_t = E[word_t]                                  embedding lookup
h_t = tanh( x_t · Wxh  +  h_(t-1) · Whh  +  b )  new memory
scores_t = h_t · Why + by                        one score per vocabulary word
p_t = softmax(scores_t)                          probabilities of the NEXT word
```

- **Loss:** cross-entropy on the true next word at every timestep, `L = -mean(log p(correct word))`.
- **Perplexity** `= exp(L)`: "the model is as uncertain as choosing among exp(L) words."
- **Training:** backpropagation through time. Softmax + cross-entropy gives the clean output error `(p − onehot) / (N·T)`, then gradients flow backward through every timestep. Weight gradients are *summed* over timesteps because the weights are shared.
- **Stability:** Adam optimizer plus global gradient-norm clipping (max 5.0).

### 3. Personal memory (`memory.py`)

A SQLite table `bigram(prev, nxt, n, last)` counts how often you typed word B right after word A. Each count decays with age:

```
effective_count = n × 0.5^(age / half_life)      half_life = 30 days
```

Every `learn()` call commits to disk immediately, so nothing is lost if the app crashes.

### 4. Blending (`SmartKeyboard.suggest`)

```
lam   = n / (n + K)                              n = your evidence for this context
score = (1 − lam) × p_rnn  +  lam × p_personal
```

With no personal evidence, `lam = 0` and the RNN decides. At `n = K` the two count equally. As you type more, `lam → 1` and your own habits win. The personal side can also suggest words the RNN has never seen.

### 5. Learning on send

1. The message's word pairs are written to `memory.db`.
2. If fine-tuning is on: one small SGD step (`lr = 1e-3`, gradient clipped to 1.0) on the RNN weights, then the model is saved. The step is deliberately tiny, because a large update on one sentence makes the model forget general language.

---

## Training data

**The quality of the suggestions is mostly the quality and amount of your text.**

- **Your own writing is best:** exported chats, emails you wrote, notes. Remove other people's private information and anything sensitive (passwords, addresses, account numbers) first.
- **Public-domain books** (for example, plain-text books from Project Gutenberg; check each book's license yourself and remove the license header and footer). Book language is more formal than texting.
- **How much:** at least a few hundred thousand words for decent results; a million or more is better. A few thousand words will overfit.
- **Format:** plain text saved as **UTF-8**, named `corpus.txt`, in the same folder as the scripts.

> If you add or change `corpus.txt` after a first run, delete `keyboard_data/model.npz` and `keyboard_data/vocab.json` to retrain. **Keep `memory.db`** unless you want to erase what the keyboard learned from you.

---

## Tuning

| Setting | Where | Default | Effect |
|---|---|---|---|
| `E_dim` / `H` | `WordRNN.__init__` | 32 / 64 | Model size: capacity vs speed and phone-friendliness |
| `max_size` | `build_vocab` | 4000 | Vocabulary size; output layer cost is `H × V` |
| `steps`, `batch`, `T`, `lr` | `train_base` | 3000, 64, 10, 3e-3 | Base training length, window length, learning rate |
| `ctx_len` | `SmartKeyboard` | 12 | How many previous words are fed to the RNN when suggesting |
| `K` | `SmartKeyboard` | 3.0 | How fast personal habits take over (smaller = faster) |
| `half_life_days` | `PersonalMemory` | 30 | How quickly old habits fade |
| fine-tune `lr` | `learn_from_text` | 1e-3 | Strength of on-device RNN adaptation |

---

## Verify and evaluate

**Gradient check first** (before any training). The math is only trustworthy if this passes:

```python
import numpy as np
from WordRNN import WordRNN
from common import gradcheck

rng = np.random.default_rng(0)
m = WordRNN(V=20, E_dim=4, H=5, seed=1)
ids, tg = rng.integers(0, 20, (3, 6)), rng.integers(0, 20, (3, 6))
m.loss(ids, tg); g = m.backward(tg)
gradcheck(lambda: m.loss(ids, tg), m.params, g)   # want error < 1e-5 on all six params
```

**Training log:** every 200 steps you get `train_loss`, `val_ppl` (validation perplexity), and `val_top3` (how often the true next word was among the 3 suggestions, the metric a keyboard actually cares about). The validation set is the **last 10% of the text** (a chronological split), and the vocabulary is built from training text only, to avoid leakage.

**Suggested experiments for your lab report**

1. **Baseline:** build a plain bigram model (counts only, no RNN) and compare top-3 accuracy. If the RNN doesn't beat it, that's a real finding.
2. **Learning curves:** plot train loss and validation perplexity per 200 steps; spot overfitting where validation turns upward.
3. **Context length:** top-3 accuracy vs `T` (3, 5, 10, 20). Does the RNN use older words?
4. **Model size sweep:** `H` = 32, 64, 128, 256 vs accuracy and training time.
5. **Personalization:** teach it 20 sentences in a particular style; measure top-3 hit rate before vs after, and vary `K`.
6. **Forgetting check:** after many fine-tune sends, re-run validation perplexity on held-out text and compare it with the original. A rise means the model is drifting from general English.
7. **Gradient norms:** log the value `clip_grads` returns per step to see whether clipping is triggering.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: corpus.txt` | Not an error with the current code; it falls back to the toy corpus. To use real text, put `corpus.txt` next to the scripts |
| `AttributeError: 'tuple' object has no attribute 'get'` | Use `self.itos, self.stoi = build_vocab(...)` (two names; the function returns two values) |
| `AttributeError: ... 'clip_grads'` on `Adam` | `clip_grads` is a standalone function: `from common import clip_grads`, then `clip_grads(g, 5.0)` |
| Crash on the *second* run in `_load` | Parenthesis bug: `WordRNN(V, E, params["Whh"].shape[0])` |
| `ImportError: make_toy_corpus` | Move `make_toy_corpus` to module level in `SmartKeyboard.py` (outside `if __name__ == '__main__':`) |
| Training ignores a new `corpus.txt` | The saved model is loaded instead of retrained; delete `model.npz` and `vocab.json` |
| `don’t` split into `don` and `t` | Normalize curly quotes: `text.lower().replace("’", "'")` in `tokenize_sentences` |
| Suggestions look random | Too little training text, or too few steps; check `val_ppl` is falling |
| `suggest() error: ...` in the status bar | Paste the message and traceback to find the failing line |
| Random crash while sampling windows | Corpus is too short (fewer words than the window length `T`) |

---

## Limitations

- Lowercase letters only: no punctuation, emoji, numbers, or multiple languages.
- The RNN's vocabulary is fixed at base training. A brand-new word is learned by personal memory (as a string) but remains `<unk>` to the RNN. Real keyboards add a character-level or subword model for this.
- Context resets with every message; only the current text box content is used.
- A single-sentence fine-tune step can slowly erode general quality; keep the learning rate small or leave the checkbox off.
- Everything runs on the UI thread. Fine at this model size; move training to a background thread for much larger models.
- This is a learning project, not a production keyboard.

---

## Privacy notes

- Everything stays on your machine: `memory.db` and `model.npz` are plain files in `keyboard_data/`.
- They contain **what you typed** (as word pairs) and weights influenced by it. Treat them as personal data and don't share them.
- A real keyboard must not learn from password fields, payment forms, or other sensitive inputs. Add that check before using this anywhere beyond experiments.
- To erase your learned habits, delete `keyboard_data/memory.db`. To fully reset, delete the whole `keyboard_data/` folder.

---

## Ideas for next steps

- Replace the vanilla RNN with the **LSTM** from your Stage 3 and compare perplexity and top-3 accuracy.
- Add a **character-level fallback** so unknown words can be completed.
- Add **trigram** personal counts for better phrase memory.
- Plot the training curves with matplotlib and add them to your lab report.
- Add a **forget / reset** button and a "learn only from messages I choose" option in the UI.
- Quantize the weights (float16 or int8) and measure the size/accuracy trade-off for phone deployment.

---

## What this project teaches

Sequence modeling, next-token prediction, embeddings, softmax with cross-entropy, backpropagation through time, gradient clipping, perplexity as a metric, on-device personalization, persistent storage, and blending a learned model with user statistics.