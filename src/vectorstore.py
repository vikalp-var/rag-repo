import os 
import faiss
import numpy as np
import pickle
from typing import List,Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingGenerator

class FaissVectorStore:
    """A simple FAISS vector store for storing and retrieving document embeddings."""
   
    def __init__(self,presist_dir: str ="faiss_store",embedding_model: str="all-MiniLM-L6-v2",chunk_size: int=1000,chunk_overlap: int=200):
        self.persist_dir = presist_dir
        os.makedirs(self.persist_dir,exist_ok=True)
        self.index=None
        self.embedding_model = embedding_model
        self.model= SentenceTransformer(embedding_model)
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        print(f"Initialized FaissVectorStore with model {embedding_model}")

    def build_from_documents(self,documents: List[str]):
        """Builds the FAISS index from the provided documents."""
        emb_pipe= EmbeddingGenerator(model_name=self.embedding_model,chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        chunks= emb_pipe.chunk_documents(documents)
        embeddings= emb_pipe.embed_chunks(chunks)
        metadatas= [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"),metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")
    
    def add_embeddings(self,embeddings: np.ndarray,metadatas: List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            print(f"[INFO] Created new FAISS index with dimension {dim}")
            self.index.add(embeddings)
            if metadatas:
                self.metadatas = metadatas
                print(f"[INFO] Added {embeddings.shape[0]} vector to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")    
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        print(f"[INFO] FAISS index and metadata saved to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")    
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadatas = pickle.load(f)
        print(f"[INFO] FAISS index and metadata loaded from {self.persist_dir}")

    def search(self,query_embedding: np.ndarray,top_k: int=5):
        D, I = self.index.search(query_embedding, top_k)  
        results=[]
        for idx, dist in zip(I[0], D[0]):
           meta = self.metadatas[idx] if idx < len(self.metadatas) else None
           results.append({"index":idx,"distance":dist,"metadata":meta})
        return results
    
    def query(self,query_text: str,top_k: int=5):
        print(f"[INFO] Generating embedding for query: {query_text}")
        query_emb= self.model.encode([query_text]).astype("float32")
        return self.search(query_emb,top_k=top_k)
    


