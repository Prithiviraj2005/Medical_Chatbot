# src/rag_pipeline.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .ingest import load_documents
from .embedder import get_embedder, embed_texts
from .indexer import build_faiss_index
from .generator import generate_answer_from_contexts  # ✅ Local/OpenAI LLM generator


# ----------------------------
# 🧠 TEXT CHUNKING FUNCTION
# ----------------------------
def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks for better retrieval context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ----------------------------
# 🧩 STEP 1: BUILD & SAVE INDEX
# ----------------------------
def build_index():
    """Loads all documents, chunks them, embeds, and saves FAISS index."""
    print("Loading documents...")
    docs = load_documents()
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["content"])
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    print("Generating embeddings...")
    model = get_embedder()
    embeddings = embed_texts(all_chunks, model)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    # Ensure vector_store directory exists
    os.makedirs("vector_store", exist_ok=True)

    # Save FAISS index and text chunks
    faiss.write_index(index, "vector_store/faiss_index.bin")
    with open("vector_store/text_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Index built and saved successfully with {len(all_chunks)} chunks.")


# ----------------------------
# 🧩 STEP 2: LOAD VECTOR STORE
# ----------------------------
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_vector_store():
    """Load FAISS index and text chunks from disk."""
    if not os.path.exists("vector_store/faiss_index.bin"):
        raise FileNotFoundError("FAISS index not found! Please run build_index() first.")
    if not os.path.exists("vector_store/text_chunks.json"):
        raise FileNotFoundError("Text chunks file missing! Please run build_index() first.")

    index = faiss.read_index("vector_store/faiss_index.bin")
    with open("vector_store/text_chunks.json", "r", encoding="utf-8") as f:
        text_chunks = json.load(f)
    return index, text_chunks


# ----------------------------
# 🧩 STEP 3: RETRIEVE + GENERATE ANSWER
# ----------------------------
def retrieve_and_answer(query, top_k=3, use_llm=True):
    """
    Retrieve top_k relevant chunks and generate a short, WHO-style accurate answer.
    """
    print(f"Processing query: {query}")
    index, text_chunks = load_vector_store()

    # Encode query
    query_vec = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_vec, dtype="float32"), top_k)

    contexts = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]

    if not contexts:
        return {"answer": "No relevant information found.", "contexts": []}

    # Use LLM for concise factual synthesis
    if use_llm:
        system_prompt = (
            "You are a precise medical assistant. "
            "Use WHO/CDC-style medical language. "
            "Give accurate, verified summaries, not assumptions."
        )
        # ✅ Fixed: removed 'question=' keyword
        answer = generate_answer_from_contexts(
    question=query,
    contexts=contexts,
    max_new_tokens=150,
    temperature=0.2
)

    else:
        combined = " ".join(contexts)
        answer = " ".join(combined.split(". ")[:2])  # fallback extractive answer

    return {
        "answer": answer.strip(),
        "contexts": contexts
    }


# ----------------------------
# 🧩 STEP 4: TEST MODE (CLI)
# ----------------------------
if __name__ == "__main__":
    print("Running RAG pipeline test...")

    # Step 1: Build index (only if missing)
    if not os.path.exists("vector_store/faiss_index.bin"):
        print("No index found. Building new index...")
        build_index()

    # Step 2: Ask a sample question
    query = "When to give Tdap booster?"
    result = retrieve_and_answer(query, top_k=3, use_llm=True)

    print("\nAnswer:")
    print(result["answer"])

    print("\nContexts Used:")
    for i, ctx in enumerate(result["contexts"], 1):
        print(f"{i}. {ctx[:200]}...")
