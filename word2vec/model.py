import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding

class Word2Vec(keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name='w2v_embedding'
        )
        self.context_embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns + 1,
        )

    def call(self, pair):
        target, context = pair

        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)

        return dots