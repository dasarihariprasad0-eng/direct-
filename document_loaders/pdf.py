from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("GRU.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()

