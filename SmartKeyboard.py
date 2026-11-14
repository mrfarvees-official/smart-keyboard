import os, json, re
import numpy as np
from memory import PersonalMemory
from WordRNN import WordRNN, evaluate
from libs import tokenize_sentences, build_vocab, encode, START
from common import Adam, gradcheck

def _atomic_write(path, writer):
    """Write to a temp file, then rename. A crash or power loss mid-write
    can never leave a half-written model (os.replace is atomic)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)
    
class SmartKeyboard:
    def __init__(self, data_dir="keyboard_data", ctx_len=12, K=3.0):
        os.makedirs(data_dir, exist_ok=True)
        self.model_path = os.path.join(data_dir, "model.npz")
        self.vocab_path = os.path.join(data_dir, "vocab.json")
        self.memory = PersonalMemory(os.path.join(data_dir, "memory.db"))
        self.ctx_len, self.K = ctx_len, K
        self.model = self.itos = self.stoi = None
        self._load()                # restores the previous session, if any
    
    # ----------- persistence ----------
    def save(self):
        _atomic_write(self.model_path, lambda f: np.savez(f, **self.model.params))
        _atomic_write(self.vocab_path, lambda f: f.write(json.dumps(self.itos).encode()))
        
    def _load(self):
        if not (os.path.exists(self.model_path) and os.path.exists(self.vocab_path)):
            return
        
        with np.load(self.model_path) as d:
            params = {k: d[k] for k in d.files}
            
        with open(self.vocab_path) as f:
            self.itos = json.load(f)
            
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        V, E = params["E"].shape
        self.model = WordRNN(V, E, params["Whh"].shape[0])
        self.model.params = params
        
    # ------------ base training (done once, offline) ------------
    def train_base(self, text, steps=3000, batch=64, T=10, lr=3e-3, seed=0):
        words = tokenize_sentences(text)
        split = int(len(words) * 0.9)       # CHRONOLOGICAL split: last 10% is validation
        self.itos, self.stoi = build_vocab(words[:split])  # vocab from TRAIN only
        ids = encode(words, self.stoi)
        tr, va = ids[:split], ids[split:]
        self.model = WordRNN(len(self.itos))
        opt, rng = Adam(self.model.params, lr=lr), np.random.default_rng(seed)
        for step in range(steps):
            s = rng.integers(0, len(tr) - T - 1, batch)         # random windows
            x = np.stack([tr[i:i + T] for i in s])
            y = np.stack([tr[i + 1:i + T + 1] for i in s])      # shifted by one word
            loss = self.model.loss(x, y)
            g = self.model.backward(y)
            Adam.clip_grads(g, 5.0)                             # stops exploding gradients
            opt.step(self.model.params, g)
            if step % 200 == 0:
                ppl, acc = evaluate(self.model, va, T)
                print(f"step {step:5d} tran_loss {loss:.3f} val_ppl {ppl:7.1f} val_top3 {acc:.3f}")
        self.save()
        
    # ------------ learning from the user (on-device, permanent) ---------------
    def learn_from_text(self, text, finetune=False, lr=1e-3):
        words = tokenize_sentences(text)
        self.memory.learn(words)
        if finetune and len(words) > 2:
            # One small SGD step on the RNN weights. Keep lr tiny: a big step on
            # one sentence make the model forget general English
            ids = encode(words[-33:], self.stoi)
            x, y = ids[:-1][None, :], ids[1:][None, :]
            self.model.loss(x, y)
            g = self.model.backward(y)
            Adam.clip_grads(g, 1.0)
            for k in g:
                self.model.params[k] -= lr * g[k]               # plain SGD: p <- p - lr * grad
            self.save()
            
    # ------------- the keystroke path ------------
    def suggest(self, text, k=3):
        text = text.lower()
        sent = re.split(r"[.!?\n]", text)[-1]                                           # only the current sentence
        words = re.findall(r"[a-z']+", sent)
        typing_word = bool(sent) and re.match(r"[a-z']", sent[-1]) is not None
        prefix = words.pop() if (typing_word and words) else "" # partial word, e.g. "ar"
        context = [START] + words[-self.ctx_len:]
        
        # (a) RNN distribution over words, restricted to the typed prefix, renormalized:
        #     p_rnn(w | context, prefix) = p(w | context) / sum of p over words starting with prefix
        p = self.model.next_probs(encode(context, self.stoi)).copy()
        p[0] = p[1] = 0.0
        if prefix:
            p *= np.fromiter(
                (w.startswith(prefix) for w in self.itos), 
                dtype=bool, 
                count=len(self.itos)
            )
            
        if p.sum() > 0:
            p /= p.sum()
            
        # (b) personal distribution: p_pers(w) = count(w) / sum of counts (under the prefix)
        counts, total_all = self.memory.next_counts(context[-1], prefix)
        z = sum(counts.values())
        
        # (c) Blend: score = (1-lam) * p_rnn + lam * p_pers, lam = n / (n + k).
        #     n = how much evidence YOU have give for this context. With none, lam = 0
        #     and the RNN decides. At n = K, lam = 0.5. As n grows, lam -> 1 and your
        #     habits win. K controls how fast the keyboard trust you.
        lam = total_all / (total_all + self.K)
        scores = {}
        for i in np.argsort(-p)[:50]:
            if p[i] > 0:
                scores[self.itos[i]] = (1 - lam) * p[i]
        
        for w, c in counts.items():                             # your words, incl. once
            scores[w] = scores.get(w, 0.0) + lam * c / z        # the RNN has never seen
        return sorted(scores, key=scores.get, reverse=True)[:k]


BASE = os.path.dirname(os.path.abspath(__file__))               # folder containing SmartKeyboard.py
CORPUS = os.path.join(BASE, "corpus.txt")
    
def make_toy_corpus(n_sentences=4000, seed=0):
        """Templated sentences so you can smoke-test the pipeline in a minute.
        Real suggestions need real text (see below)."""
        import random
        rng = random.Random(seed)
        places = ["gym", "office", "park", "station", "cafe", "library", "airport"]
        times = ["tomorrow", "tonight", "today", "later", "soon"]
        things = ["meeting", "movie", "class", "dinner", "game"]
        templates = [
            "how are you", "how are you doing {t}", "see you at the {p} {t}",
            "i am going to the {p} {t}", "what time is the {x}",
            "are you coming to the {x} {t}", "thank you so much",
            "let me know when you are at the {p}", "i will call you {t}",
        ]
        out = [rng.choice(templates).format(p=rng.choice(places),
                                            t=rng.choice(times),
                                            x=rng.choice(things))
            for _ in range(n_sentences)]
        return ". ".join(out) + "."
    
if __name__ == '__main__':
    # --- gradient check (you already passed this) ---
    # ... keep your gradcheck code here ...

    kb = SmartKeyboard(os.path.join(BASE, "keyboard_data"))
    if kb.model is None:
        if os.path.exists(CORPUS):
            with open(CORPUS, encoding="utf-8") as f:     # utf-8 matters on Windows
                text = f.read()
        else:
            print("corpus.txt not found, training on a TOY corpus (templated sentences).")
            text = make_toy_corpus()
        kb.train_base(text, steps=1500)    # toy data learns fast; use 3000+ for real text

    print(kb.suggest("how are "))
    print(kb.suggest("see you at the "))
    print(kb.suggest("how ar"))
    kb.learn_from_text("see you at the gym tomorrow.")
    print(kb.suggest("see you at the "))