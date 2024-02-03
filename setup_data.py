import tensorflow as tf 
from tensorflow import keras
from keras import layers


def load_data(path, max_vocab, seq_len, batch_size):
    data = tf.data.TextLineDataset(path)

    # Remove empty lines
    non_empty_ln = []
    for line in data:
        if tf.strings.length(line) > 0:
            non_empty_ln.append(line)
    
    data = tf.data.Dataset.from_tensor_slices(non_empty_ln)

    # Normalize, Split, and Map strings
    vectorize_layer = layers.TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=max_vocab,
        output_mode='int',
        output_sequence_length=seq_len
    )

    # Adapt to create vocabulary
    vectorize_layer.adapt(data.batch(1024))
    inv_vocab = vectorize_layer.get_vocabulary()

    # Batching Data for Performance
    ''' DEPRECATED
    data_v = data.apply(
        tf.data.experimental.map_and_batch(
            vectorize_layer, 
            batch_size=batch_size,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    )
    '''
    data_v = (
        data
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(vectorize_layer, num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch()
    )

    sequences = list(data_v.as_numpy_iterator())
    
    # for seq in sequences[:5]:
    #     print(f"{seq} => {[inv_vocab[i] for i in seq]}")

    return sequences, inv_vocab

