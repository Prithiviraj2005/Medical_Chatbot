# src/ingest.py

import os
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ fixed import

DATA_DIR = Path("data")

# ----------------------------
# 🧹 Text Cleaning
# ----------------------------
def clean_text(text: str) -> str:
    """Basic cleaning: remove extra whitespace and newlines."""
    return " ".join(text.split())

# ----------------------------
# 📄 Load Text Files
# ----------------------------
def load_text_files() -> List[Dict]:
    """Load all .txt files from the data directory."""
    items = []
    for file in DATA_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = clean_text(f.read())
        items.append({"source": file.name, "content": content})
    return items

# ----------------------------
# 📘 Load PDFs
# ----------------------------
def load_pdfs() -> List[Dict]:
    """Load all .pdf files from the data directory."""
    items = []
    for file in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        text = clean_text(text)
        items.append({"source": file.name, "content": text})
    return items

# ----------------------------
# 📚 Combine All Documents
# ----------------------------
def load_documents() -> List[Dict]:
    """Load all text and PDF documents."""
    docs = []
    docs.extend(load_text_files())
    docs.extend(load_pdfs())
    print(f"📂 Loaded {len(docs)} documents from {DATA_DIR}")
    return docs

# ----------------------------
# 🧩 FIX: Add prepare_documents()
# ----------------------------
def prepare_documents() -> List[Dict]:
    """Prepare documents for FAISS indexing (adds IDs and text fields)."""
    docs = load_documents()
    prepared_docs = []
    for i, doc in enumerate(docs):
        prepared_docs.append({
            "id": i,
            "source": doc.get("source", "unknown"),
            "text": doc.get("content", "")
        })
    print(f"✅ Prepared {len(prepared_docs)} documents for indexing.")
    return prepared_docs

# ----------------------------
# 🔍 Test Run
# ----------------------------
if __name__ == "__main__":
    print("🚀 Testing document ingestion...\n")
    docs = load_documents()
    print(f"✅ Successfully loaded {len(docs)} documents.\n")

    for i, doc in enumerate(docs[:2], 1):  # show first 2 previews
        print(f"{i}. {doc['source']}")
        print(doc['content'][:300], "...\n")
