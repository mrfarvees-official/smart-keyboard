import json, os, re, sqlite3, time
import numpy as np
from common import Adam, gradcheck, softmax

UNK, START = "<unk>", "<s>" #ids 0 and 1

def tokenize_sentences(text):
    """Lowercase, split into sentences, put a <s> marker before each one.
    The marker lets the model learn which word START sentences ("how", "i", "the")."""
    out = []
    for s in re.split(r"[.!?\n]+", text.lower()):
        words = re.findall(r"[a-z']+", s)    # letters + apostrophes (don't, it's)
        if words:
            out += [START] + words
    return out

def build_vocab(words, max_size=4000):
    """Keep the most frequent words; everything else becomes <unk>.
    The softmax has one output per vocabulary word, so vocab size directly
    sets the cost: output layer = H x V weights."""
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
        
    top = sorted(
        (w for w in counts if w not in (UNK, START)), 
        key=counts.get, 
        reverse=True
    )[:max_size -2] 
    
    itos = [UNK, START] + top
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi

def encode(words, stoi):
    return np.array([stoi.get(w, 0) for w in words], dtype=np.int64) # unknown -> 0

if __name__ == '__main__':
    ts = tokenize_sentences("It's a beautiful day!.")
    bv = build_vocab(ts)
    en = encode(ts, bv[1])
    
    print(ts)
    print(bv)
    print(en)