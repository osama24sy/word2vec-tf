from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow import keras
import numpy as np
from attention import NextWord

from next_word import next_word, next_word_avg, next_word_basic


# Load the model from the .model file
embeddings = KeyedVectors.load("/content/drive/MyDrive/NLP Projects/text8_word2vec.model")
vocab_size = len(embeddings.wv.index_to_key)
word_embeddings = tf.zeros((vocab_size, 100)).numpy()

# Fill the tensor by accessing word vectors
for i, word in enumerate(embeddings.wv.index_to_key):
    embedding = embeddings.wv[word]
    word_embeddings[i] = embedding

word_embeddings = tf.convert_to_tensor(word_embeddings)

print(next_word_avg("The oldest city in the world is", embeddings))
print(next_word_basic("The oldest city in the world is", embeddings=embeddings, topn=5))

# Generate training data
import gensim.downloader as api

data = api.load("text8")
dataset = []
labels = []
i = 0
for line in data:
  sequences = np.array_split(line, 625)
  for seq in sequences:
    tok = []
    lab = []
    for word in seq:
      try:
        tok.append(embeddings.wv[word])
        lab.append(embeddings.wv.key_to_index[word])
      except:
        tok = []
        break
    # print(tok)
    if len(tok) != 0:
      dataset.append(tok[:-1])
      labels.append(lab[1:]) # make indices
  i += 1
  if i == 100: break
dataset = tf.constant(dataset)
labels = tf.constant(labels)

# Create the model
model = NextWord(71290, 100, word_embeddings)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x=dataset,
    y=labels,
    epochs=5, 
    batch_size=512,  
)

# Plot the training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()

# Test the model
print(next_word("The oldest city in the world is",
          embeddings,
          model, k = 15))

# Save the model
model.save("attention.keras")