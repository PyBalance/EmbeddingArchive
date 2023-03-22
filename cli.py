from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader,UnstructuredMarkdownLoader, UnstructuredHTMLLoader,UnstructuredPowerPointLoader, SRTLoader,UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

# load a pdf file and create a FAISS embedding file
def load_file(filename):
    try:
        if filename.endswith('.pdf'):
            return PyPDFLoader(filename)
        elif filename.endswith('.docx'):
            return UnstructuredWordDocumentLoader(filename)
        elif filename.endswith('.md'):
            return UnstructuredMarkdownLoader(filename)
        elif filename.endswith('.html'):
            return UnstructuredHTMLLoader(filename)
        elif filename.endswith('.pptx'):
            return UnstructuredPowerPointLoader(filename)
        elif filename.endswith('.srt'):
            return SRTLoader(filename)
        else:
            return UnstructuredFileLoader(filename)
    except Exception as e:
        print(f"Error: {e}")
        
def ingest_docs(loader, path):
    """Get documents from web pages."""
    raw_documents = loader.load()
    print(raw_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)

    # Save vectorstore
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)
        
if __name__ == "__main__":
    # Continue with the rest of the code here
    #loader=load_file('test/ruihuayanjiu.pdf')
    #ingest_docs(loader,"test/ruihuayanjiu.pkl")
    with open("test/ruihuayanjiu.pkl",'rb') as file:
        db = pickle.load(file)
    while True:
        query = input("Enter a query (type 'exit' to quit): ")
        if query == 'exit':
            break
        docs = db.similarity_search_with_score(query)
        for doc in docs:
            print(doc)
