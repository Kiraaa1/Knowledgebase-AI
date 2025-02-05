import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the document
with open("sample_doc_2.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split into chunks
chunks = text.split("\n\n")

# Convert chunks into embeddings
embeddings = model.encode(chunks)

# Store in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save FAISS index
faiss.write_index(index, "vector_store.index")

# Save chunks for retrieval
import pickle
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)