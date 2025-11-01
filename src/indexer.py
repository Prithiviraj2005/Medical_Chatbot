# src/indexer.py

import os
import json
import numpy as np
import faiss
from pathlib import Path  # ✅ Added this import

# ✅ Import document loader
from .ingest import load_documents  

# ✅ Import utility functions
from .utils import get_embedder, embed_texts

# ------------------------------
# 📂 Directory setup
# ------------------------------
VECTOR_DIR = Path('vector_store')
VECTOR_DIR.mkdir(exist_ok=True)

# ------------------------------
# 📄 File paths
# ------------------------------
META_FILE = VECTOR_DIR / 'meta.json'
INDEX_FILE = VECTOR_DIR / 'faiss_index.bin'

# ------------------------------
# 🧠 Build or Load FAISS Index
# ------------------------------
def build_or_load_index(rebuild=False, batch_size=5000):
    """Build or load FAISS index from documents."""
    print("🚀 Starting index build or load process...")

    # Step 1: Load documents
    docs = load_documents()
    texts = [d['content'] for d in docs]
    metas = [{'source': d['source'], 'content': d['content']} for d in docs]

    # Step 2: Load embedding model
    model = get_embedder()
    all_embs = []

    # Step 3: Create embeddings in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"⚙️ Embedding batch {i // batch_size + 1}/{len(texts) // batch_size + 1}...")
        emb = embed_texts(batch, model)
        all_embs.append(emb)

    # Step 4: Combine embeddings
    if all_embs:
        embeddings = np.vstack(all_embs)
    else:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()))

    # Step 5: Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    # Step 6: Save index and metadata
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved FAISS index ({len(metas)} vectors) to {INDEX_FILE}")
    return index, metas

# ------------------------------
# 📦 Load FAISS Index
# ------------------------------
def load_index():
    """Load FAISS index and metadata from disk."""
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("❌ Index or metadata missing. Run build_or_load_index(rebuild=True).")

    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, 'r', encoding='utf-8') as f:
        metas = json.load(f)

    print(f"✅ Loaded index with {len(metas)} entries.")
    return index, metas

# ------------------------------
# 🧩 Alias for compatibility
# ------------------------------
def build_faiss_index(rebuild=False, batch_size=5000):
    """Alias for backward compatibility."""
    return build_or_load_index(rebuild=rebuild, batch_size=batch_size)


# ------------------------------
# 🧪 Manual test entry point
# ------------------------------
if __name__ == "__main__":
    print("🧠 Running FAISS index builder test...\n")
    build_or_load_index(rebuild=True)
