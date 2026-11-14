import numpy as np
from common import softmax

class WordRNN:
    def __init__(self, V, E_dim=32, H=64, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            # E: embedding table. Row i is a learned E_dim-vector for word i.
            # Words used in similar contexts end up with similar vectors,
            # which is how the model generalizes "gym" ~ "pool" ~ "office".
            "E":    rng.normal(0, 0.1, (V, E_dim)),
            "Wxh":  rng.normal(0, np.sqrt(1.0 / E_dim), (E_dim, H)),    # input -> memory
            "Whh":  rng.normal(0, np.sqrt(1.0 / H), (H, H)),            # old memory -> new memory
            "bh":   np.zeros(H),
            "Why":  rng.normal(0, np.sqrt(1.0 / H), (H, V)),            # memory -> one score per word
            "by":   np.zeros(V),
        }
        self.cache = None
        self.probs = None
        
    def forward(self, ids):
        """ids: (N, T) word ids. Returns logits (N, T, V): a score for every 
        vocabulary word at every timestamp."""
        P = self.params
        N, T = ids.shape
        H, V = P["Whh"].shape[0], P["by"].shape[0]
        hs, xs = [np.zeros((N, H))], []             # hs[t] = memory BEFORE reading word t
        logits = np.empty((N, T, V))
        for t in range(T):
            x = P["E"][ids[:, t]]                   # look up embeddings: (N, E_dim)
            # h_t = tanh(x_t Wxh + h_{t-1} Whh + b): mix the new word with old memory
            h = np.tanh(x @ P["Wxh"] + hs[-1] @ P["Whh"] + P["bh"])
            # logits_t = h_t Why + by: "how well each word fit as the NEXT word?"
            logits[:, t, :] = h @ P["Why"] + P["by"]
            xs.append(x)
            hs.append(h)
        self.cache = (ids, xs, hs)
        return logits
    
    def loss(self, ids, targets):
        """targets[n, t] = the word that actually come after ids[n, t].
        Cross-entropy: L = -mean( log p(correct next word) ), averaged over N*T steps.
        Perplexity = exp(L) = "the model is as unsure as choosing among exp(L) words"."""
        logits = self.forward(ids)
        # softmax: p_i = exp(s_i) / sum_j exp(s_j), scores -> probabilities that sum to 1
        probs = softmax(logits)
        self.probs = probs
        N, T = targets.shape
        picked = probs[np.arange(N)[:, None], np.arange(T)[None, :], targets]
        return -np.mean(np.log(picked + 1e-12))     # +1e-12 guards log(0)
    
    def backward(self, targets):
        """Backprop through time. Call right after loss()."""
        P = self.params
        ids, xs, hs = self.cache
        N, T = ids.shape
        g = {k: np.zeros_like(v) for k, v in P.items()}
        
        # Softmax + cross-entropy: dL/dlogits = (probs - onehot(target)) / (N*T)
        # Same clean cancellation as sigmoid + BCE in Stage 1.
        dlogits = self.probs.copy()
        dlogits[np.arange(N)[:, None], np.arange(T)[None, :], targets] -= 1.0
        dlogits /= (N * T)
        
        dh_next = np.zeros_like(hs[0])      # error flowing back from the FUTURE timestamp
        for t in reversed(range(T)):
            dl = dlogits[:, t, :]
            # Output layer at step t: weights are shared across steps, so += accumulates.
            g["Why"] += hs[t + 1].T @ dl    # (H,N)@(N,V) -> (H,V)
            g["by"] += dl.sum(axis=0)
            # h_t gets error from TWO places: this step's prediction AND the future steps.
            dh = dl @ P["Why"].T + dh_next
            dz = dh * (1.0 - hs[t + 1] ** 2)
            g["Wxh"] += xs[t].T @ dz
            g["Whh"] += hs[t].T @ dz
            g["bh"] += dz.sum(axis=0)
            # Gradient into the embedding rows that were looked up at step t.
            # np.add.at (not g["E"][ids] += ...) because the same word can
            # appear twice in the batch and both gradients must be ADDED.
            np.add.at(g["E"], ids[:, t], dz @ P["Wxh"].T)
            dh_next = dz @ P["Whh"].T       # pass error one step further back
        return g
    
    def next_probs(self, ctx_ids):
        """Probability of every word being next, given a context. Used at typing time."""
        logits = self.forward(np.array([ctx_ids]))
        return softmax(logits[0, -1])
    
def evaluate(mode, ids, T=10, max_windows=200):
    """Held-out perplexity and top-3 accuracy (the metric a keyboard cares about:
    was the real next word among the 3 suggestions?)."""
    n = min((len(ids) -1) // T, max_windows)
    X = ids[:n * T].reshape(n, T)
    Y = ids[1:n * T + 1].reshape(n, T)
    loss = mode.loss(X, Y)
    top3 = np.argpartition(-mode.probs, 3, axis=-1)[..., :3]
    return float(np.exp(loss)), float((top3 == Y[..., None]).any(axis=-1).mean())