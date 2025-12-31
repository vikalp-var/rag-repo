from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingGenerator:
    """Generates embeddings for a list of documents using a pre-trained model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.chunk_size = chunk_size
        self.chunk_overlap= chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Splits documents into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generates embeddings for the given chunks."""
        texts= [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    