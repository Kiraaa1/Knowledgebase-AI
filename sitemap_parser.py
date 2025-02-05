# sitemap_parser.py

import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def parse_sitemap(sitemap_path):
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    ns = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    urls = []
    for url in root.findall("s:url", ns):
        loc = url.find("s:loc", ns)
        if loc is not None and loc.text:
            urls.append(loc.text.strip())
    return urls

def fetch_document_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator="\n")
    # Clean and collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_text(text, chunk_size=500):
    # Split text into words, then recombine into chunks
    words = re.split(r'\s+', text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_vector_store(sitemap_path, model_name="all-MiniLM-L6-v2"):
    print("Parsing sitemap...")
    urls = parse_sitemap(sitemap_path)
    print(f"Found {len(urls)} URLs.")
    
    all_chunks = []
    for url in urls:
        print(f"Processing: {url}")
        text = fetch_document_text(url)
        if text:
            chunks = chunk_text(text, chunk_size=500)
            all_chunks.extend(chunks)
    
    print(f"Total text chunks: {len(all_chunks)}")
    
    # Create embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and chunks
    faiss.write_index(index, "vector_store.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    
    print("Vector store and chunks saved.")

if __name__ == "__main__":
    sitemap_file = "sitemap.xml"  # Adjust the path if needed
    build_vector_store(sitemap_file)
