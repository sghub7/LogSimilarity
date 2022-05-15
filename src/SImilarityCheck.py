from scipy import spatial
from sent2vec.vectorizer import Vectorizer


sentences = [
    "This is an awesome book to learn NLP.",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
]

vectorizer = Vectorizer()
vectorizer.run(sentences)
vectors_bert = vectorizer.vectors

dist_1 = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])
dist_2 = spatial.distance.cosine(vectors_bert[0], vectors_bert[2])
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))

# dist_1: 0.043, dist_2: 0.192