import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Your Personal Study Assistant",
    page_icon="📚"
)

st.title("📚 Study Assistant")
st.write("Upload your study material or add a website and ask questions.")

# Create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Create Chroma database
vector_store = Chroma(
    persist_directory="chroma_database",
    embedding_function=embedding_model
)

# Create language model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Add PDF documents
st.sidebar.header("📚 Add Study Material")

files = st.sidebar.file_uploader(
    "Upload PDF books / notes",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Add Documents"):

    if files:

        for file in files:

            try:
                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp:

                    temp.write(file.read())
                    path = temp.name

                # Load PDF
                loader = PyPDFLoader(path)
                docs = loader.load()

                # Split PDF into chunks
                chunks = splitter.split_documents(docs)

                # Add filename information to each chunk
                for chunk in chunks:
                    chunk.metadata["source_type"] = "pdf"
                    chunk.metadata["file_name"] = file.name

                # Store chunks in Chroma
                vector_store.add_documents(chunks)

                # Delete temporary PDF
                os.remove(path)

                st.sidebar.success(
                    f"{file.name} added!"
                )

            except Exception as e:

                st.sidebar.error(
                    f"Error adding {file.name}: {e}"
                )

    else:

        st.sidebar.warning(
            "Upload a PDF first."
        )


# Add website
st.sidebar.subheader("🌐 Add Website")

url = st.sidebar.text_input(
    "Website URL"
)

if st.sidebar.button("Add Website"):

    if url:

        try:
            # Load website
            loader = WebBaseLoader(url)
            docs = loader.load()

            # Split website content
            chunks = splitter.split_documents(docs)

            # Add website information to each chunk
            for chunk in chunks:
                chunk.metadata["source_type"] = "website"
                chunk.metadata["file_name"] = url

            # Store website chunks in Chroma
            vector_store.add_documents(chunks)

            st.sidebar.success(
                "Website added!"
            )

        except Exception as e:

            st.sidebar.error(
                f"Error: {e}"
            )

    else:

        st.sidebar.warning(
            "Enter a URL."
        )


# Delete stored documents
st.sidebar.subheader("🗑️ Delete Study Material")

# Get all stored data
all_data = vector_store.get()

documents = {}

# Get filenames from metadata
if all_data["metadatas"]:

    for metadata in all_data["metadatas"]:

        if not metadata:
            continue

        file_name = metadata.get("file_name")
        source_type = metadata.get("source_type", "unknown")

        if file_name:
            documents[file_name] = source_type


# Show stored documents
if documents:

    document_options = []

    for file_name, source_type in documents.items():

        if source_type == "pdf":
            display_name = f"📄 PDF - {file_name}"

        elif source_type == "website":
            display_name = f"🌐 Website - {file_name}"

        else:
            display_name = f"📁 {file_name}"

        document_options.append(display_name)

    selected_document = st.sidebar.selectbox(
        "Select material to delete",
        document_options
    )

    # Get original filename or URL
    selected_name = selected_document.split(
        " - ",
        1
    )[1]

    if st.sidebar.button("Delete Selected"):

        try:
            # Find chunks belonging to selected document
            data = vector_store.get(
                where={
                    "file_name": selected_name
                }
            )

            ids_to_delete = data["ids"]

            if ids_to_delete:

                # Delete selected document chunks
                vector_store.delete(
                    ids=ids_to_delete
                )

                st.sidebar.success(
                    f"{selected_name} deleted!"
                )

                # Refresh the application
                st.rerun()

            else:

                st.sidebar.warning(
                    "Document not found."
                )

        except Exception as e:

            st.sidebar.error(
                f"Delete error: {e}"
            )

else:

    st.sidebar.info(
        "No study material stored."
    )


# Ask questions
query = st.chat_input(
    "Ask something about your study material..."
)

if query:

    # Display user question
    st.chat_message("user").write(query)

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 7,
            "lambda_mult": 0.5
        }
    )

    # Search for relevant documents
    docs = retriever.invoke(query)

    # Create context from retrieved documents
    if docs:

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

    else:

        context = "No relevant information found."

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful study assistant.

            Answer only from the provided context.

            Do not use your own knowledge.

            If the answer is not present in the context,
            say:

            I could not find the answer
            in the provided documents.
            """
        ),
        (
            "human",
            """
            Context:

            {context}

            Question:

            {question}
            """
        )
    ])

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Get answer from LLM
    response = llm.invoke(final_prompt)

    # Display answer
    st.chat_message("assistant").write(
        response.content
    )