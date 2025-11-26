import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, vocab=None):
        default_vocab = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.vocab = list(vocab or default_vocab)
        self.stoi = {ch: i+1 for i, ch in enumerate(self.vocab)}
        self.itos = {i+1: ch for i, ch in enumerate(self.vocab)}
        self.pad_id = 0

    def encode(self, text, max_len):
        ids = [self.stoi.get(ch.lower(), self.stoi.get(" ", 1)) for ch in text]
        return ids[:max_len] + [self.pad_id] * (max_len - len(ids))
    

    def decode(self, ids):
        return "".join([self.itos.get(i, " ") for i in ids if i != self.pad_id])
    
    def save(self, path):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, mode='w') as f:
            json.dump({"vocab": self.vocab})

    
    @staticmethod
    def load(path):
        with open(path, "r") as f:
            data = json.load(f)
        return CharTokenizer(vocab=data["vocab"])


        
