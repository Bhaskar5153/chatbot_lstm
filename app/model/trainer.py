class Trainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def fit(self, X, Y, epochs=5, batch_size=16):
        self.model.model.fit([X, Y], Y epochs=epochs, batch_size=batch_size, validate_splits=0.1)

    def save_all(self, weights_path, tokenizer_path):
        self.model.save_weights(weights_path)
        self.tokenizer.save(tokenizer_path)

    