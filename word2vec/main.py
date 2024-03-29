import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from word2vec.setup_data import load_data
from word2vec.generate_training_data import generate_training_data_tf
from word2vec.model import Word2Vec
from word2vec.embeddings import save_embeddings

# HYPERPARAMETERS
VOCAB_SIZE = 15000
SEQ_LEN = 20
WINDOW_SIZE = 3
NEG_SAMP = 3
SEED = 42
BATCH_SIZE = 1024
BUFFER_SIZE = 1000
EMBEDDING_DIM = 300

# Prepare The dataset
sequences, vocab = load_data(
    path="data/new_text8.txt",
    max_vocab=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE
)

targets, contexts, labels = generate_training_data_tf(
    sequences=sequences,
    window_size=WINDOW_SIZE,
    num_ns=NEG_SAMP,
    vocab_size=VOCAB_SIZE,
    seed=SEED
)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

print(dataset)

word2vec = Word2Vec(VOCAB_SIZE, EMBEDDING_DIM, NEG_SAMP)
word2vec.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

word2vec.fit(dataset, epochs=15, callbacks=[tensorboard_callback])

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

save_embeddings(vocab, weights)