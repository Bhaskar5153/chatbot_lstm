import tensorflow as tf 
import numbers as np
import pathlib
import json

class LSTMModel:
    def __init__(self, vocab_size, emb_dim=128, enc_units=128, dec_units=128, max_a_len=256):
        self.model = self._build(vocab_size, emb_dim, enc_units, dec_units)
        self.max_a_len = max_a_len
        self.vocab_size = vocab_size

    def _build(self, vocab_size, emb_dim, enc_units, dec_units):
        enc_in = tf.keras.Input(shape=(None, ))
        enc_dim = tf.keras.layers.Embedding(vocab_size+1, emb_dim, mask_zero=True)(enc_in)
        enc_out, h, c = tf.keras.layers.LSTM(enc_units, return_sequence=True, return_state=True)(enc_dim)

        dec_in = tf.keras.Input(shape=(None,))
        dec_emb = tf.keras.layers.Embedding(vocab_size+1, emb_dim, mask_zero=True)(dec_in)
        dec_out = tf.keras.layers.LSTM(enc_units, return_sequence=True, return_state=True)(dec_emb, initial_state=[h, c])
        logits = tf.keras.layer.Dense(vocab_size+1, activation="softmax")(dec_out)

        model = tf.keras.Model([enc_in, dec_in], logits)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        return model
    
    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    
    def greedy_decode(self, encoder_input, start_id):
        seed = np.full((encoder_input.shape[0], self.max_a_len), start_id)
        probs = self.model.predict([encoder_input, seed], verbose=0)
        return probs.argmax(axis=1)
    
        


    

        