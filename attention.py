import tensorflow as tf
from tensorflow import keras
import numpy as np

class Attention(keras.Model):
  def __init__(self, embedding_dim):
    super().__init__()
    self.qkv_layer = keras.layers.Dense(embedding_dim*3)
    self.dim = embedding_dim

  def call(self, context, mask=None, scaled=True):
    qkv = self.qkv_layer(context)
    # print(qkv.shape)
    q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

    attention_scores = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1]))

    if scaled:
      attention_scores = tf.divide(attention_scores, np.sqrt(attention_scores.shape[-1]))

    if mask is not None:
      attention_scores = tf.where(mask == 0, -np.inf, attention_scores)

    causal_mask = tf.linalg.band_part(tf.ones([context.shape[-2], context.shape[-2]]), -1, 0)
    attention_scores = tf.where(causal_mask == 0, -np.inf, attention_scores)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    attention_scores = tf.matmul(attention_scores, v) # T, d
    return attention_scores
  
class NextWord(keras.Model):
  def __init__(self, vocab_size, embed_dim, embeddings):
    super().__init__()
    self.attention = Attention(embed_dim)
    self.dim = embed_dim
    self.embeddings = embeddings

  def call(self, sequence):
    att_embed = tf.matmul(self.attention(sequence), tf.transpose(self.embeddings))
    att_embed = tf.convert_to_tensor(att_embed)
    return att_embed
