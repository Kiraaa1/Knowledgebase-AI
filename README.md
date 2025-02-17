Next-Generation AI Knowledgebase Chatbot: A RAG System
Project Summary
This project is an advanced AI-driven knowledgebase chatbot that transforms static technical documentation into an interactive, intelligent support system. Leveraging a Retrieval-Augmented Generation (RAG) framework, the chatbot dynamically retrieves and synthesizes information from extensive documentation to deliver context-aware responses in real time.

Key features include:

Document Ingestion & Processing:

Parses a sitemap XML to extract targeted URLs (e.g., pages under https://docs.dewacloud.com/docs/).
Fetches, cleans, and preprocesses technical documents.
Splits lengthy texts into manageable chunks for efficient processing.
Semantic Embeddings & Vector Search:

Utilizes Sentence Transformers (e.g., all-MiniLM-L6-v2) to generate semantic embeddings from text chunks.
Employs FAISS to perform high-speed vector similarity searches and retrieve relevant content.
Retrieval-Augmented Generation (RAG):

Combines the retrieved document context with a language model (via Ollama, e.g., TinyLlama) to generate detailed, contextually accurate answers.
Supports prompt engineering to guide the model’s output style and formatting.
Real-Time Chat Interface:

Built with Flask and SocketIO to provide an interactive, responsive user experience.
Enables seamless, real-time Q&A through a modern chat UI.
Technical Highlights
RAG Framework: Integrates document retrieval with AI response generation for enhanced, context-aware outputs.
Efficient Document Processing: Ingests and preprocesses documentation from a sitemap, ensuring robust and scalable data handling.
Scalable Semantic Search: Uses FAISS and Sentence Transformers to enable fast and precise vector-based searches.
Interactive Chat Interface: Delivers real-time responses through a Flask and SocketIO-powered chat application.
Modular Architecture: Easily extendable for cloud-based or local deployments, and adaptable to multilingual documentation (e.g., Bahasa Indonesia).
Repository Structure
sitemap_parser.py:
Parses the sitemap, fetches and processes documents, chunks text, and builds a FAISS vector store with embeddings.

rag_gen.py:
Loads the vector store and text chunks, retrieves relevant content based on user queries, and generates AI responses using a language model.

flask_app.py:
Serves a real-time chat interface via Flask and SocketIO, connecting the frontend with the AI backend.

templates/index.html:
Contains the chat interface UI code.

