import io
import numpy as np
import tensorflow as tf

def save_embeddings(vocab, weights):
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


def import_vectors_and_words(vectors_file, words_file):
    # Load vectors from TSV file
    with open(vectors_file, 'r') as f:
        vectors = np.loadtxt(f, delimiter='\t')

    # Load words from TSV file
    with open(words_file, 'r') as f:
        words = f.read().splitlines()

    # Ensure the number of vectors and words matches
    if len(vectors) != len(words):
        raise ValueError("Number of vectors and words must match.")

    # Create the dictionary
    word_vectors = dict(zip(words, vectors))

    return word_vectors

def cosine_similarity(vector1, vector2):

    # Ensure vectors have the same dimensions
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same dimensions.")

    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes (L2 norms)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return cosine_similarity

def similar_words(word_vectors, target_word, n=5):

    if target_word not in word_vectors:
        return []  # Handle missing target word

    target_vector = word_vectors[target_word]
    similarities = {word: cosine_similarity(target_vector, word_vectors[word])
                     for word in word_vectors if word != target_word}

    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:n]

