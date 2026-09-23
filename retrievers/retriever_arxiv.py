import arxiv
from langchain_community.retrievers import ArxivRetriever

# we create a retriever to retrieve data
retriever = ArxivRetriever(
    load_max_docs = 3,  # no.of docs to retrieve
    load_all_available_meta= True
)
# query arxiv
docs = retriever.invoke("quantum computing")

for i, doc in enumerate(docs):
    print(f"document:{i+1}")
    print("title",doc.metadata.get('title'))
    print("summary",doc.page_content[:500])
