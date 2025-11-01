import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts, model):
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
