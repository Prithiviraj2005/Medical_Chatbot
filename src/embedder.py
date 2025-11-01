from sentence_transformers import SentenceTransformer

def get_embedder(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
