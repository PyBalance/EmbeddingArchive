from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader,UnstructuredMarkdownLoader, UnstructuredHTMLLoader,UnstructuredPowerPointLoader, SRTLoader,Unstructured File Loader
from langchain.embeddings import HuggingFaceHubEmbeddings

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
            raise ValueError("Invalid file type")
    except Exception as e:
        print(f"Error: {e}")


loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content)
repo_id = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceHubEmbeddings(
    repo_id=repo_id,
    task="feature-extraction",
    huggingfacehub_api_token="my-api-key",
)