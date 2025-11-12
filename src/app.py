import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectorDB import VectorDB
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and Groq LLM.
    """

    def __init__(self, file_type: str = "pdf"):
        """Initialize the RAG assistant."""
        self.file_type = file_type.lower()
        self.llm = self._initialize_llm()
        self.vector_db = VectorDB()

        # Strict prompt to avoid hallucination
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use ONLY the context provided below to answer the question.
If the context does not contain the answer, reply exactly with:
"I’m sorry, but the information you’re asking for is not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

        # Build the chain: prompt → LLM → output parser
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("✅ RAG Assistant initialized successfully (Groq Model Ready)")

    def _initialize_llm(self):
        """Initialize Groq model using the API key."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")

        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        print(f"🤖 Using Groq model: {model_name}")

        return ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.0  # Strict factual responses only
        )

    def load_and_add_documents(self, data_dir: str = "../data") -> None:
        """
        Load PDF or TXT files and add them to the vector database.
        """
        # Adjust path if script runs from src/
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_data_dir = os.path.normpath(os.path.join(base_path, data_dir))

        if not os.path.exists(full_data_dir):
            print("⚠️ Data folder not found. Please create a 'data/' folder and add documents.")
            return

        documents = self.vector_db.load_documents(data_dir=full_data_dir, file_type=self.file_type)

        if not documents:
            print(f"⚠️ No {self.file_type.upper()} files found in {full_data_dir}.")
            return

        print(f"📂 Loaded {len(documents)} {self.file_type.upper()} documents from '{data_dir}'")

        # Print document info safely for both dict and Document types
        for doc in documents:
            if isinstance(doc, dict):
                content = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
            else:
                content = getattr(doc, "page_content", "")
                metadata = getattr(doc, "metadata", {})

            content_len = len(content.strip())
            print(f"📄 Loaded: {os.path.basename(metadata.get('source', 'unknown'))} | {content_len} characters")

        print(f"📚 Adding {len(documents)} {self.file_type.upper()} documents to the vector DB...")
        self.vector_db.add_documents(documents)

    def invoke(self, query: str, n_results: int = 3) -> str:
        """Query the RAG assistant and get context-based answers."""
        search_results = self.vector_db.search(query, n_results=n_results)
        context_chunks = search_results.get("documents", [])
        context_text = "\n\n".join(context_chunks)

        if not context_text.strip():
            return "I’m sorry, but the information you’re asking for is not available in the provided documents."

        response = self.chain.invoke({"context": context_text, "question": query})
        return response


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        print("🚀 Welcome to the RAG Assistant (Groq + ChromaDB)")
        print("Choose which file type to load:")
        print("1. PDF files")
        print("2. TXT files")

        choice = input("Enter 1 or 2: ").strip()
        file_type = "pdf" if choice == "1" else "txt"

        # Initialize the assistant
        assistant = RAGAssistant(file_type=file_type)

        # Load and add documents
        assistant.load_and_add_documents()

        while True:
            user_input = input("\n🧑‍💻 Ask a question (or type 'quit' to exit): ").strip()
            if user_input.lower() == "quit":
                print("👋 Exiting RAG assistant. Goodbye!")
                break

            answer = assistant.invoke(user_input)
            print("\n🤖 Answer:\n", answer)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your .env file contains:")
        print("GROQ_API_KEY=<your_groq_api_key>")
        print("GROQ_MODEL=llama-3.1-8b-instant (or any supported Groq model)")


if __name__ == "__main__":
    main()
