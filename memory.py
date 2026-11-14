import sqlite3
import time

from libs import START

class PersonalMemory:
    """Counts how often YOU type word B right after A. Stored on disk
    committed after every learn(), so it survives app restarts and crashes."""
    
    def __init__(self, path, half_life_days=30.0):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS bigram(
            prev TEXT, nxt TEXT, n REAL, last REAL, PRIMARY KEY(prev, nxt))""")
        self.db.commit()
        self.half_life = half_life_days * 86400.0
        
    def learn(self, words):
        now = time.time()
        for a, b in zip(words, words[1:]):
            if b == START:
                continue                    # <s> is a maker, never a suggestion
            # Upsert: insert with n=1, or add 1 if the pair exists (needs SQLite >= 3.24)
            self.db.execute(
                "INSERT INTO bigram VALUES(?,?,1,?) "
                "ON CONFLICT(prev, nxt) DO UPDATE SET n=n+1, last=?", (a, b, now, now)
            )
            self.db.commit()                # <- this commit is the "permanent" part
    
    def next_counts(self, prev, prefix=""):
        """Returns ({word: effective_count} for words starting with `prefix`,
        total effective count over ALL words seen after `prev`)."""
        now = time.time()
        counts, total = {}, 0.0
        for nxt, n, last in self.db.execute("SELECT nxt, n, last FROM bigram WHERE prev=?", (prev,)):
            # Recency decay: effective = n * 0.5^(age / half_life).
            # After one half-life unused, a pair counts half as much, so
            # old habits fade and new once take over.
            c = n * 0.5 ** ((now - last) / self.half_life)
            total += c
            if nxt.startswith(prefix):
                counts[nxt] = c
        return counts, total