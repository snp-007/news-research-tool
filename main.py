import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if process_url_clicked:

    urls = [url for url in urls if url.strip()]

    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    # Faster loader than UnstructuredURLLoader
    loader = WebBaseLoader(urls)

    start = time.time()
    main_placeholder.text("Loading articles... ⏳")

    data = loader.load()

    main_placeholder.text(
        f"Articles loaded in {time.time() - start:.2f} sec ✅"
    )

    # Split documents
    start = time.time()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ","]
    )

    docs = text_splitter.split_documents(data)

    main_placeholder.text(
        f"Created {len(docs)} chunks in {time.time() - start:.2f} sec ✅"
    )

    # Create embeddings and vector store
    start = time.time()

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    main_placeholder.text(
        f"Vector store built in {time.time() - start:.2f} sec ✅"
    )

    # Save vector store
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text("Processing complete! 🎉")

# Question Input
query = st.text_input("Question:")

if query:

    if not os.path.exists(file_path):
        st.warning("Please process URLs first.")
        st.stop()

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
    )

    result = chain(
        {"question": query},
        return_only_outputs=True
    )

    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")

    if sources:
        st.subheader("Sources")
        for source in sources.split("\n"):
            if source.strip():
                st.write(source)