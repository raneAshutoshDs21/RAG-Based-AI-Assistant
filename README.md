# RAG-Based-AI-Assistant
A Retrieval-Augmented Generation (RAG) app using LangChain, ChromaDB, Sentence Transformers, and Groq’s LLM to answer questions strictly from your PDF or text documents—ensuring factual, document-grounded responses with zero hallucination.

## Project Structure
```
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
```

## Setup Instructions (VS Code Terminal)

1️⃣ Clone the repository
git clone https://github.com/raneAshutoshDs21/RAG-Based-AI-Assistant.git
cd RAG-Based-AI-Assistant

2️⃣ Create a virtual environment
python -m venv .venv

3️⃣ Activate the virtual environment

Windows PowerShell

.venv\Scripts\activate


macOS/Linux

source .venv/bin/activate

4️⃣ Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

5️⃣ Install dependencies
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file inside your project root and add:

GROQ_API_KEY=your_groq_api_key_here


⚠️ Make sure .env is included in .gitignore (to keep your API key private).

🚀 Running the Project

Run the app directly from the terminal inside VS Code:

python src/app.py


If you’re using Streamlit UI, then run:

streamlit run src/app.py

🧩 How It Works

Document Loading:
All .txt or .pdf files from the data/ folder are read.

Chunking & Embedding:
Each document is split into smaller chunks and embedded using the SentenceTransformer model (all-MiniLM-L6-v2).

Vector Database (ChromaDB):
The embeddings are stored locally inside a Chroma collection called rag_documents.

Retrieval + Generation:
When a user asks a question, the app:

Retrieves the most relevant chunks from ChromaDB.

Passes them to the Groq LLM for context-aware answers.

Ensures the model responds strictly based on the loaded documents.

🧰 Tech Stack
Component	Purpose
LangChain	Framework for chaining LLM and retriever
ChromaDB	Local vector store for document embeddings
SentenceTransformers	Generates semantic embeddings for documents
Groq API	Provides fast, low-latency LLM inference
Python 3.10+	Core programming language
dotenv	Secure environment variable management
🧠 Example Query

After running the app:

python src/app.py


Example conversation:

🧑‍💻 Ask a question (or type 'quit' to exit): What is artificial intelligence?

🤖 Answer:

## Project Working

<img width="1538" height="789" alt="Screenshot 2025-11-11 191522" src="https://github.com/user-attachments/assets/c2f18d5a-5b15-477e-9646-1946196e8c21" />



🧾 License

This project is licensed under the MIT License
.
