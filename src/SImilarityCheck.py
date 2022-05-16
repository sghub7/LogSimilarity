from scipy import spatial
from sent2vec.vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

pDict={}

sentences = [
    "s3 connection timed out ",
    "This is an amazing NLP model.",
    "We can interchangeably use embedding, encoding, or vectorizing.",
    "My name is Bond.. James Bond",
    "this is same exception ie s3 connection timed out but with more words ",
    "s3 throttling exception "
]

## Similarity with Sentence2Vec
vectorizer = Vectorizer()
vectorizer.run(sentences)
vectors_bert = vectorizer.vectors

x=cosine_similarity(
    [vectors_bert[0]],
    vectors_bert[1:]
)

print("Target Sentence -",sentences[0])
sentences_1=sentences[1:]
pDict["sentences"]=sentences_1
models =["Sentence2Vec-distilbert-base-uncased","SentenceTransformer-Hugging Face-bert-base-nli-mean-tokens","Tensorflow-UniversalSentenceEncoder"]
# pDict["models"]=models
pDict["Sentence2Vec-distilbert-base-uncased"]=[]
pDict["SentenceTransformer-HuggingFace-bert-base-nli-mean-tokens"]=[]
pDict["Tensorflow-UniversalSentenceEncoder"]=[]
pDict["Average"]=[]



print("**** Similarity with Sentence2Vec -distilbert-base-uncased Model ")

for i in range(0,len(sentences_1)):
    pDict["Sentence2Vec-distilbert-base-uncased"].append(x[0][i])
    # print(f"{sentences_1[i]} :: Similarity Score = {x[0][i]}")

# print(pDict["Sentence2Vec-distilbert-base-uncased"])
# new = pd.DataFrame.from_dict(pDict)
# print(new.head())

### Sentence Transformer.. Hugging Face Model
hf_model=SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = hf_model.encode(sentences)
y=cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
print("Target Sentence -",sentences[0])
print("**** Similarity with SentenceTransformer- Hugging Face  -bert-base-nli-mean-tokens Model ")
for i in range(0,len(sentences_1)):
    pDict["SentenceTransformer-HuggingFace-bert-base-nli-mean-tokens"].append(y[0][i])
    # print(f"{sentences_1[i]} :: Similarity Score = {y[0][i]}")

print(pDict["SentenceTransformer-HuggingFace-bert-base-nli-mean-tokens"])
# new = pd.DataFrame.from_dict(pDict)
# print(new.head())


### TensorFlow
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
module_url = "saved_model.pb"
# import tarfile
# tar = tarfile.open(module_url, "r:gz")
# tar.extractall()
# tar.close()

model = tf.keras.models.load_model('models')
print ("module %s loaded" % module_url)
tf_sentence_embeddings = model(sentences)
z=cosine_similarity(
    [tf_sentence_embeddings[0]],
    tf_sentence_embeddings[1:]
)
print("Target Sentence -",sentences[0])
print("**** Similarity with Tensorflow  -Universal Sentence Encoder Model ")
for i in range(0,len(sentences_1)):
    pDict["Tensorflow-UniversalSentenceEncoder"].append(z[0][i])


print("**** Average Similarity Scores*****")
for i in range(0,len(sentences_1)):
    pDict["Average"].append((x[0][i] + y[0][i] + z[0][i])/3)
    # print(f"{sentences_1[i]} :: Similarity Score = {(x[0][i] + y[0][i] + z[0][i])/3 }")
print(pDict["Average"])

new = pd.DataFrame.from_dict(pDict)
print(new.head())



"""
Target Sentence - s3 connection timed out 
**** Similarity with Sentence2Vec -distilbert-base-uncased Model 
This is an amazing NLP model. :: Similarity Score = 0.9462440013885498
We can interchangeably use embedding, encoding, or vectorizing. :: Similarity Score = 0.7585642337799072
My name is Bond.. James Bond :: Similarity Score = 0.9174299240112305
this is same exception ie s3 connection timed out but with more words  :: Similarity Score = 0.8607273101806641
s3 throttling exception  :: Similarity Score = 0.9919077157974243
Target Sentence - s3 connection timed out 
**** Similarity with SentenceTransformer- Hugging Face  -bert-base-nli-mean-tokens Model 
This is an amazing NLP model. :: Similarity Score = 0.6209895014762878
We can interchangeably use embedding, encoding, or vectorizing. :: Similarity Score = 0.502221941947937
My name is Bond.. James Bond :: Similarity Score = 0.41913869976997375
this is same exception ie s3 connection timed out but with more words  :: Similarity Score = 0.7557125091552734
s3 throttling exception  :: Similarity Score = 0.6477135419845581
module saved_model.pb loaded
Target Sentence - s3 connection timed out 
**** Similarity with Tensorflow  -Universal Sentence Encoder Model 
[[-0.00086745  0.05307842  0.11314272  0.60867417  0.6328026 ]]
This is an amazing NLP model. :: Similarity Score = -0.0008674515411257744
We can interchangeably use embedding, encoding, or vectorizing. :: Similarity Score = 0.05307842046022415
My name is Bond.. James Bond :: Similarity Score = 0.1131427213549614
this is same exception ie s3 connection timed out but with more words  :: Similarity Score = 0.608674168586731
s3 throttling exception  :: Similarity Score = 0.6328026056289673
**** Average Similarity Scores*****
This is an amazing NLP model. :: Similarity Score = 0.5221220254898071
We can interchangeably use embedding, encoding, or vectorizing. :: Similarity Score = 0.4379548629124959
My name is Bond.. James Bond :: Similarity Score = 0.4832371075948079
this is same exception ie s3 connection timed out but with more words  :: Similarity Score = 0.7417046229044596
s3 throttling exception  :: Similarity Score = 0.7574745814005533

                                           sentences  Sentence2Vec-distilbert-base-uncased  SentenceTransformer-HuggingFace-bert-base-nli-mean-tokens  Tensorflow-UniversalSentenceEncoder   Average
0                      This is an amazing NLP model.                              0.946244                                           0.620990                                    -0.000867  0.522122
1  We can interchangeably use embedding, encoding...                              0.758564                                           0.502222                                     0.053078  0.437955
2                       My name is Bond.. James Bond                              0.917430                                           0.419139                                     0.113143  0.483237
3  this is same exception ie s3 connection timed ...                              0.860727                                           0.755713                                     0.608674  0.741705
4                           s3 throttling exception                               0.991908                                           0.647714                                     0.632803  0.757475

"""