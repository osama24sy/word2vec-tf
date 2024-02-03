from gensim.models import KeyedVectors
from next_word import next_word

embeddings = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

print(next_word("The capital of the united states is", embeddings=embeddings, topn=20))
