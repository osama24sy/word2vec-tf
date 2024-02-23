import tensorflow as tf
from tensorflow import keras
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def next_word_avg(sentence, embeddings, topn=10) :
  tokens = word_tokenize(sentence.lower())
  sequence = np.array([embeddings.wv[word] for word in tokens])
  print(sequence.shape)
  avg = np.sum(sequence, axis=0) / sequence.shape[0]
  output = embeddings.wv.most_similar(positive=[avg], topn=topn)
  return output

def next_word_basic(sentence, embeddings, topn=10) :
    tokens = word_tokenize(sentence.lower())
    stop_words = stopwords.words('english')
    words = [token for token in tokens if token not in stop_words]

    context = np.array([embeddings[word] for word in words[:-1]]) # (T, d)
    query = np.array([embeddings[words[-1]]]) # (d)

    query = tf.squeeze(query, axis=0) #  (d, 1)

    output = context * query # (T, d)

    output = tf.reduce_sum(output, axis=1, keepdims=True) # (T, 1)

    soft = tf.transpose(tf.nn.softmax(tf.transpose(output))) # (T, 1)

    soft = tf.tile(soft, [1, tf.shape(context)[1]]) # (T, d)

    last = soft * context # (T, d)

    last = tf.reduce_sum(last, axis=0) # (d,)

    last = last.numpy()

    output = embeddings.most_similar(positive=[last.numpy()], topn=topn)

    return output

def next_word(sent, embeddings, model, k = 1):
    tokens = word_tokenize(sent.lower())
    stop_words = stopwords.words('english')
    words = [token for token in tokens if token not in stop_words]
    context = tf.constant([embeddings.wv[word] for word in words])
    context = tf.expand_dims(context, axis=0)
    logits = model(context)
    print(logits[:, -1, 0])
    # lm_head = tf.nn.softmax(logits, axis = -1)
    # TODO: Random sampling for word selection
    results = tf.math.top_k(logits[:, -1, :], k).indices.numpy()
    # return results[-1]
    return [embeddings.wv.index_to_key[result] for result in results[-1]]
