import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silences TensorFlow info/warning logs, shows only errors
import tensorflow_hub as hub
import tensorflow as tf

import numpy as np

# Sample dataset
sentences = [
    "I love machine learning",
    "Machine learning is fun",
    "Deep learning models are powerful",
    "I enjoy coding in Python",
    "Python is my favorite language",
    "AI is the future",
    "Learning never stops",
    "I like playing football",
    "Football is my favorite sport",
    "I love reading books"
]

print("Loading Univarsal Encoder...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Model Loded!...")

sentence_embaddings = embed(sentences)
# print("sentence_embaddings",sentence_embaddings)

#cosine similarity function
def cosine_similarity(vec1,vec2):
    dot = np.dot(vec1,vec2)
    norm = np.linalg.norm(vec1)*np.linalg.norm(vec2)
    return dot/norm
#top k similar sentance
def most_similar(query:str,top_k=3):
    query_emb = embed([query])[0].numpy()
    sims = [cosine_similarity(query_emb,s.numpy()) for s in sentence_embaddings]
    top_indices = np.argsort(sims)[::-1][:top_k]
    result = [(sentences[i],float(sims[i])) for i in top_indices]
    return result
