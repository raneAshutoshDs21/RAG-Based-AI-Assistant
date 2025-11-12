import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with SentenceTransformer embeddings.
    Supports PDF and TXT files.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"🔹 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create Chroma collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"✅ Vector database initialized with collection: {self.collection_name}")

    # --------------------------------------------------------------------
    # 📘 File loading section
    # --------------------------------------------------------------------
    def load_documents(self, data_dir: str, file_type: str = "pdf") -> List[Dict[str, Any]]:
        """
        Load all PDF or TXT documents from a directory.
        Args:
            data_dir: Path to folder containing files.
            file_type: 'pdf' or 'txt'.
        Returns:
            List of documents in {"content": text, "metadata": {...}} format.
        """
        documents = []

        for file in os.listdir(data_dir):
            if file_type.lower() == "pdf" and file.endswith(".pdf"):
                path = os.path.join(data_dir, file)
                text = self._extract_text_from_pdf(path)
                documents.append({"content": text, "metadata": {"source": file}})

            elif file_type.lower() == "txt" and file.endswith(".txt"):
                path = os.path.join(data_dir, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                documents.append({"content": text, "metadata": {"source": file}})

        print(f"📂 Loaded {len(documents)} {file_type.upper()} documents from '{data_dir}'")
        return documents

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2.
        """
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"⚠️ Error reading PDF {file_path}: {e}")
        return text.strip()

    # --------------------------------------------------------------------
    # 🧩 Embedding & storage section
    # --------------------------------------------------------------------
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into smaller chunks for better embedding.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to Chroma vector DB.
        Each document should be a dict with keys: 'content' and optional 'metadata'.
        """
        if not documents:
            print("⚠️ No documents to add.")
            return

        print(f"🧠 Processing {len(documents)} documents for embedding...")

        all_texts, all_metadatas, all_ids = [], [], []

        for doc_index, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_text(content)
            print(f"🧩 Document {doc_index+1} split into {len(chunks)} chunks.")

            for i, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_index}_chunk{i}"
                all_texts.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)

        embeddings = self.embedding_model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

        self.collection.add(
            ids=all_ids,
            embeddings=embeddings.tolist(),
            documents=all_texts,
            metadatas=all_metadatas,
        )

        print(f"✅ Successfully added {len(all_texts)} chunks to ChromaDB.")

    # --------------------------------------------------------------------
    # 🔍 Search section
    # --------------------------------------------------------------------
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Perform similarity search against stored documents.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        if not results or not results.get("documents"):
            print("⚠️ No similar documents found.")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }
