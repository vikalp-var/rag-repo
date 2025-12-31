from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch 
from src.embedding import EmbeddingGenerator


#example usage

if __name__ == "__main__":
    data_directory = "data"
    documents = load_all_documents(data_directory)


    # chunks = EmbeddingGenerator().chunk_documents(documents)
    # chunkvectors = EmbeddingGenerator().embed_chunks(chunks)
    # print(chunkvectors)
    # print(f"Total documents loaded: {documents}")

    store= FaissVectorStore('faiss_store')
    # store.build_from_documents(documents)
    store.load()
    # print(store.query("what is react",top_k=3))
    rag_search = RAGSearch()
    # query = "what is jsx"
    query = input("Enter your question: ")
    summary = rag_search.search_and_summarize(query, top_k=3)
    print(f"Summary Response:\n{summary}")
