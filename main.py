"""1.here we use groq model so we import the ChatGroq
2.we use the HuggingFaceEmbeddings to create embeddings for the documents
3.so here we use the Chroma vector store to store the embedding we create for the documents
4.we use ChatPromptTemplate to give the llm instructions """
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
 # we use the huugingface embedding to create the embeddings
embedding_model = HuggingFaceEmbeddings()

# here we store the embeddings in vectorstore
vector_store = Chroma(
    persist_directory = "chroma_database",
    embedding_function = embedding_model
)

# here we retriever takes the query and returns the most relevant documents from a the vectorstore bsaed on the search type 
#example of search type like MMR(max marginal relevance)
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":3, # we get three relevant documents from retriever
        "fetch_k":7, # we search top 7 relevnant embeddings
        "lambda_mult": 0.5
    }
)

# here we use the modle "openai/gpt-oss-20b"
llm = ChatGroq(model = "openai/gpt-oss-20b")

# here we define the instructions to llm 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """you are a helpful assistant that answers questions based on the provided context,
        if the answer not present in the context,
        say : i could not find the answer in the provided document
        """),
        ("human", """
        context:{context}
        Question:{question}"""
         
         )
    ]
)

print("the rag system is create")

print("press 0 to exit")

# here we create an infinite loop
while True:
    query = input("you:")
    if query == "0":
        break
    docs = retriever.invoke(query)
    context = "\n\n".join(
    [doc.page_content for doc in docs])

    final_prompt = prompt.invoke(
        {
        "context" : context,
        "question":query
        }
    )

    response = llm.invoke(final_prompt)
    print(f"\n AI response: {response.content}")

