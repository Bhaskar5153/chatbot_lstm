import numpy as np

class QADataset:
    def __init__(self, tokenizer, max_q_len=128, max_a_len=256):
        self.tokinizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

    
    def build_arrays(self, qa_pairs):
        X = [self.tokinizer.encode(q, self.max_q_len) for q, _ in qa_pairs]
        Y = [self.tokinizer.encode(a, self.max_a_len) for _, a in qa_pairs]
        return np.array(X), np.array(Y)
    

    