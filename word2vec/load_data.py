# import nltk
import time

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def tokenizer(text, sent_len) :
    start = time.time() # Start timer
    tokenized_text = text.split()
    words = set(tokenized_text)
    data = []

    for i in range(0, len(tokenized_text), sent_len):
        sentence = []
        for j in range(i, min(i + sent_len, len(tokenized_text))):
            sentence.append(tokenized_text[j])
        data.append(sentence)


    end = time.time() # End timer

    info = {
        "wordCount": len(tokenized_text),
        "sentenceCount": len(data),
        "dictSize": len(words),
        "runtime": end - start
    }
    return data, words, info

def load_data(path, sent_len) :
    # Import file
    corpus_raw = ""
    text_data = open(path, "r", encoding='unicode_escape')
    for line in text_data.readlines():
        corpus_raw += line
    text_data.close()
    
    print(f"The data has been successfully loaded from {path}")
    
    corpus_raw = corpus_raw.lower()

    data, words, info = tokenizer(corpus_raw, sent_len)

    print(f"{path} has been successfully processed and tokenized\n",
          f"Total number of words: {info['wordCount']}\n",
          f"Total sentence count: {info['sentenceCount']}\n",
          f"The dictionary size is : {info['dictSize']}\n", 
          f"The runtime is {info['runtime']} seconds\n")

    return corpus_raw, data, words, info
