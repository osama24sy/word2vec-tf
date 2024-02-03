import tensorflow as tf
from tensorflow import keras
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def next_word(sentence, embeddings, topn=10) :
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