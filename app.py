import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="GP Trainee Virtual Assistant", page_icon="🩺")

st.markdown(
    """
    <div style="text-align:center; padding:12px 0 6px;">
        <div style="font-size:42px; line-height:1.1;">🩺 GP Trainee Virtual Assistant</div>
        <div style="color:#4a5568; margin-top:6px;">
            Ask about study leave, exception reports, mileage claims, and more.
        </div>
    </div>
    <hr style="margin:12px 0 18px;">
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    st.stop()

DOCUMENTS_FOLDER = "documents"


def get_files_signature(folder_path: str) -> str:
    """Changes whenever PDFs are added/removed/edited."""
    if not os.path.isdir(folder_path):
        return ""
    sigs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            stat = os.stat(path)
            sigs.append(f"{filename}:{stat.st_size}:{stat.st_mtime_ns}")
    sigs.sort()
    return "|".join(sigs)


@st.cache_resource(show_spinner="Building document database... This may take a few minutes.")
def build_vectorstore(files_signature: str):
    """Build and cache FAISS for the current PDFs."""
    if not os.path.isdir(DOCUMENTS_FOLDER):
        raise ValueError("No 'documents' folder found.")
    all_docs = []
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DOCUMENTS_FOLDER, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = filename
            all_docs.extend(docs)
    if not all_docs:
        raise ValueError("No PDFs found in the 'documents' folder.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(splits, embeddings)


files_signature = get_files_signature(DOCUMENTS_FOLDER)
vectorstore = build_vectorstore(files_signature)

prompt_template = """
You are a helpful assistant for GP trainees in the UK.
Use ONLY the following context (local guidance, policies, or documents) to answer the question.

If the answer is not clearly stated, say you are not sure and suggest who they could contact.

Always:
- Keep answers concise and practical and formatted with headings and bullet points and emojis for ease of reading.
- If there is conflicting information, say so.
- Do not make up answers.
- Do not provide legal or medical advice.
- Do not share personal opinions.
- Do not reference your own capabilities or limitations.
- Do not mention AI, language models, or similar.
- Do not provide source code or technical details.
- If explicitly asked for hypothetical examples, make them clearly hypothetical and do not answer anything unrelated to GP training.
- Do not provide contact details unless they are in the context.
- Do not include square brackets or document citations during the answer as these will be provided separately in the source documents.
- If the question is unrelated to GP training, politely decline to answer.
- If you are unsure about the answer, suggest who they could contact for help (GP Trainee Council, Training Programme Director, etc.)
- If the question is about personal situations, suggest contacting the Training Programme Director or relevant authority.
- If the question is about mental health or wellbeing, suggest contacting appropriate support services.
- If the question is about exceptions or appeals, suggest following official procedures as per the provided documents.
- If the question is about mileage claims, study leave, or other administrative processes, provide step-by-step guidance based on the documents.
- If the question is about exam preparation or assessments, provide tips based on the context.
- if the question is about career progression or opportunities, provide information based on the documents.
- If the question is about mentorship or support, suggest relevant programs mentioned in the context.
- If the question is about work-life balance, provide advice based on the context.

Question: {question}
=========
{context}
=========
Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=api_key),
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

with st.form("search_form", clear_on_submit=False):
    col1, col2 = st.columns([12, 1])
    with col1:
        # single-line input so pressing Enter submits the form
        query = st.text_input("Ask your question", placeholder="Type your question here...")
    with col2:
        # magnifying glass button
        submit = st.form_submit_button("🔍")

if submit and query:
    with st.spinner("Thinking..."):
        result = qa({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

    st.markdown("### 📘 Answer:")
    st.write(answer)

    with st.expander("🔍 Documents used"):
        for i, doc in enumerate(sources, start=1):
            filename = doc.metadata.get("source_file", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")
            st.markdown(f"**Source {i}:** `{filename}`, page {page}")
