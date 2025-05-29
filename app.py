import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="GP Trainee Chatbot")
st.title("💬 GP Trainee Support Chatbot")
st.write("Ask questions about study leave, exception reports, mileage claims, and more!")

query = st.text_input("Ask your question")

if query:
    # Load all PDFs from the 'documents' folder
    folder_path = "documents"
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            all_docs.extend(docs)

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Embed and create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(splits, embeddings)

    # QA chain setup
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=api_key),
        retriever=vectorstore.as_retriever()
    )

    # Get answer
    with st.spinner("Thinking..."):
        answer = qa.run(query)
        st.markdown("### 📘 Answer:")
        st.write(answer)


st.subheader("📚 Downloadable Files")

folder_path = "documents"

with st.expander("Show available files for download"):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"📄 Download {filename}",
                    data=f,
                    file_name=filename,
                    mime="application/pdf"
                )