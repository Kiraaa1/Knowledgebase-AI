# rag_gen.py

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama  # Or switch to your preferred API/model library

# Load the embedding model and pre-built vector store/chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_store.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def retrieve_relevant_chunks(query, top_k=2):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_llm(question):
    context = retrieve_relevant_chunks(question)
    prompt = f"Using the following documentation, answer the question:\n\n{context}\n\nQ: {question}"
    
    # Use your chosen model here. For example, using Ollama with tinyllama:
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
