import os
import re
import tkinter as tk
from tkinter import ttk

from SmartKeyboard import SmartKeyboard, make_toy_corpus
from libs import tokenize_sentences, encode

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus.txt")


def load_keyboard():
    """Load the saved model, or train the base model once if none exists."""
    kb = SmartKeyboard(os.path.join(BASE, "keyboard_data"))
    if kb.model is None:
        if os.path.exists(CORPUS):
            with open(CORPUS, encoding="utf-8") as f:
                text = f.read()
        else:
            print("corpus.txt not found, training on a TOY corpus.")
            text = make_toy_corpus()
        kb.train_base(text, steps=1500)      # prints progress in the console
    return kb


def sentence_loss(kb, text):
    """Cross-entropy of the RNN on one message: -mean(log p(actual next word)).
    Lower = the model finds this text less surprising. We measure it before
    and after learning so you can SEE the update do something."""
    words = tokenize_sentences(text)
    if len(words) <= 2:                       # <s> + 1 word is too short to score
        return None
    ids = encode(words[-33:], kb.stoi)        # same window learn_from_text trains on
    return float(kb.model.loss(ids[:-1][None, :], ids[1:][None, :]))


class KeyboardApp:
    def __init__(self, root, kb):
        self.root, self.kb = root, kb
        self.current = []                     # the suggestions currently on screen
        root.title("Smart Keyboard (RNN + personal memory)")
        root.geometry("540x540")

        # ---- chat history (read-only) ----
        self.history = tk.Text(root, state="disabled", wrap="word", height=18,
                               padx=8, pady=8, font=("Segoe UI", 11))
        self.history.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.history.tag_configure("me", justify="right", foreground="#0b57d0")

        # ---- suggestion bar: three clickable buttons ----
        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=8)
        self.sug_buttons = []
        for i in range(3):
            b = ttk.Button(bar, text="", state="disabled",
                           command=lambda i=i: self.apply_suggestion(i))
            b.pack(side="left", expand=True, fill="x", padx=2)
            self.sug_buttons.append(b)

        # ---- input row ----
        row = ttk.Frame(root)
        row.pack(fill="x", padx=8, pady=8)
        self.var = tk.StringVar()
        self.entry = ttk.Entry(row, textvariable=self.var, font=("Segoe UI", 12))
        self.entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Send", command=self.send).pack(side="left", padx=(6, 0))

        self.finetune = tk.BooleanVar(value=True)
        ttk.Checkbutton(root, variable=self.finetune,
                        text="Also fine-tune the RNN weights on each send "
                             "(off = only personal memory learns)"
                        ).pack(anchor="w", padx=8)
        self.status = ttk.Label(root, text="Type something. Tab accepts the first suggestion.",
                                foreground="#555555", wraplength=500)
        self.status.pack(fill="x", padx=8, pady=(2, 8))

        # ---- events ----
        self.entry.bind("<Return>", lambda e: self.send())
        self.entry.bind("<Tab>", self.accept_first)
        # trace fires on EVERY change of the text, whether you typed it or a
        # suggestion click inserted it, so suggestions always stay in sync.
        self.var.trace_add("write", lambda *a: self.refresh())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.entry.focus_set()
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self):
        """Ask the keyboard for suggestions for the current text."""
        try:
            self.current = self.kb.suggest(self.var.get(), k=3)
        except Exception as e:                # keep the UI alive; show what broke
            self.current = []
            self.status.config(text=f"suggest() error: {e!r}")
        for i, b in enumerate(self.sug_buttons):
            if i < len(self.current):
                b.config(text=self.current[i], state="normal")
            else:
                b.config(text="", state="disabled")

    def apply_suggestion(self, i):
        if i >= len(self.current):
            return
        word = self.current[i]
        text = re.sub(r"[A-Za-z’']+$", "", self.var.get())   # drop the half-typed word
        self.var.set(text + word + " ")                      # triggers refresh() via trace
        self.entry.focus_set()
        self.entry.icursor("end")

    def accept_first(self, event):
        self.apply_suggestion(0)
        return "break"                        # stops Tab from moving focus away

    def add_line(self, text, tag=None):
        self.history.config(state="normal")
        self.history.insert("end", text + "\n", tag)
        self.history.config(state="disabled")
        self.history.see("end")

    def send(self):
        text = self.var.get().strip()
        if not text:
            return
        self.var.set("")
        self.add_line(text, "me")

        finetune = self.finetune.get()
        before = sentence_loss(self.kb, text) if finetune else None
        # This is the "train on every send" step:
        #   1) personal memory: pair counts written to SQLite (permanent at once)
        #   2) optional: one small SGD step on the RNN weights, then model saved
        self.kb.learn_from_text(text, finetune=finetune)
        after = sentence_loss(self.kb, text) if finetune else None

        n_pairs = self.kb.memory.db.execute("SELECT COUNT(*) FROM bigram").fetchone()[0]
        msg = f"Saved to personal memory ({n_pairs} word pairs stored). "
        if before is not None and after is not None:
            msg += f"RNN loss on this message: {before:.4f} -> {after:.4f}"
        self.status.config(text=msg)

    def close(self):
        self.kb.memory.db.close()
        self.root.destroy()


if __name__ == "__main__":
    kb = load_keyboard()                      # trains in the console first run only
    root = tk.Tk()
    KeyboardApp(root, kb)
    root.mainloop()