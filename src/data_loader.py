from pathlib import Path
from typing import List , Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """Load all supported files from the data directory and convert to LangChain document structure
    Supported: PDF, TXT, CSV, DOCX, XLSX, JSON
    """
    # Use project root folder.
    data_path = Path(data_dir).resolve()
    print(f"Loading documents from: {data_path}")
    documents=[]

    #PDF files
    pdf_files= list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"Loading PDF file: {pdf_file}")
        try:
         loader = PyPDFLoader(str(pdf_file))
         loaded = loader.load()
         print(f"Loaded {len(loaded)} documents from {pdf_file}")
         documents.extend(loaded)
        except Exception as e:
         print(f"Error loading PDF file {pdf_file}: {e}")

        return documents

    #TXT files
    txt_files= list(data_path.glob("*.txt"))
    print(f"Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}") 
    for txt_file in txt_files:
        print(f"Loading TXT file: {txt_file}")
        try:
         loader = TextLoader(str(txt_file))
         loaded = loader.load()
         print(f"Loaded {len(loaded)} documents from {txt_file}")
         documents.extend(loaded)
        except Exception as e:
            print(f"Error loading TXT file {txt_file}: {e}")

    #CSV files
    csv_files= list(data_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"Loading CSV file: {csv_file}")
        try:
         loader = CSVLoader(str(csv_file))
         loaded = loader.load()
         print(f"Loaded {len(loaded)} documents from {csv_file}")
         documents.extend(loaded)
        except Exception as e:
            print(f"Error loading CSV file {csv_file}: {e}")            

    #DOCX files
    docx_files= list(data_path.glob("*.docx"))          
    print(f"Found {len(docx_files)} DOCX files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"Loading DOCX file: {docx_file}")
        try:
         loader = Docx2txtLoader(str(docx_file))
         loaded = loader.load()
         print(f"Loaded {len(loaded)} documents from {docx_file}")
         documents.extend(loaded)
        except Exception as e:
            print(f"Error loading DOCX file {docx_file}: {e}")    

