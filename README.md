# RAG-Based-AI-Assistant
A Retrieval-Augmented Generation (RAG) app using LangChain, ChromaDB, Sentence Transformers, and Groq’s LLM to answer questions strictly from your PDF or text documents—ensuring factual, document-grounded responses with zero hallucination.

## Project Structure
RAG-Based-AI-Assistant/
│
├── .venv/                        # Virtual environment (not pushed to GitHub)
│
├── data/                         # Folder for source documents
│   ├── ai.txt
│   ├── climate_change.txt
│   └── Space exploration.txt
│
├── src/                          # Main source code folder
│   ├── app.py                    # Main application (entry point)
│   └── vectorDB.py               # Handles ChromaDB setup and embeddings
│
├── .env                          # Stores environment variables like GROQ_API_KEY
│
├── .gitignore                    # Prevents .venv and .env from being tracked
│
├── LICENSE                       # License file
│
└── requirements.txt              # Python dependencies
