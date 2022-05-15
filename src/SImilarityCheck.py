from scipy import spatial
from sent2vec.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



sentences = [
    "s3 connection timed out ",
    "DistilBERT is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
    "My name is Bond.. James oBond",
    "this is same exception ie s3 connection timed out but with more words "
]

vectorizer = Vectorizer()
vectorizer.run(sentences)
vectors_bert = vectorizer.vectors

dist_1 = spatial.distance.cosine(vectors_bert[0], vectors_bert[1])
dist_2 = spatial.distance.cosine(vectors_bert[0], vectors_bert[2])
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
y=cosine_similarity(
    [vectors_bert[0]],
    vectors_bert[1:]
)

print(y)

### Sentence Transformer
hf_model=SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = hf_model.encode(sentences)
print("*****",sentence_embeddings.shape)
x=cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
print(x)


### Tensolrflow
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
tf_sentence_embeddings = model(sentences)
print("*****",tf_sentence_embeddings.shape)
x=cosine_similarity(
    [tf_sentence_embeddings[0]],
    tf_sentence_embeddings[1:]
)
print(x)