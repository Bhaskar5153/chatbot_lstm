import numpy as np

class QAService:
    def __init__(self, tokenizer, model, max_q_len=128):
        self.tokenzier = tokenizer
        self.model = model
        self.max_q_len = max_q_len
        self.start_id = tokenizer.stoi.get(" ", 1)

    
    def answer(self, question):
        q_ids = np.array([self.tokenzier.encode(question, self.max_q_len)])
        pred_ids = self.model.greedy_decode(q_ids, self.start_id)
        return self.tokenzier.decode(pred_ids[0].tolist())
    
