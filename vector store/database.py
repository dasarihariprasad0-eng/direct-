from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis it is a famous module in python.", metadata={"source": "DataScience_book"}),
    Document(page_content="rag is a framework used in artificial intelligence for providing contextual information and avoid hallucunations.", metadata={"source": "rag_book"}),
]

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding= embedding_model,
    persist_directory = "chroma_database"

)

res = vectorstore.similarity_search("what is rag framework", k = 2)
for r in res:
    print(r)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("what is rag framework")

for d in docs:
    print(d)