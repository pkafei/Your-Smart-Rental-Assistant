import streamlit as st
import os
from gpt4all import GPT4All
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure directories exist
DOCS_FOLDER = "documents/"
os.makedirs(DOCS_FOLDER, exist_ok=True)

# Load GPT4All model
model_path = os.path.abspath("models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
os.environ["GPT4ALL_MODEL_PATH"] = os.path.dirname(model_path)

try:
    local_model = GPT4All(model_path, allow_download=False)
except Exception as e:
    st.error(f"⚠️ Error loading GPT4All model: {e}")
    st.stop()

# Load documents from folder
def load_documents():
    docs = []
    for file in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Process and store docs in ChromaDB (Handles missing documents)
def process_documents():
    docs = load_documents()

    if not docs:
        st.warning("⚠️ No documents found in `documents/`. Add files before asking questions.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        st.warning("⚠️ No valid text found in documents. Ensure files have readable content.")
        return None

    # Fix ChromaDB Connection by Persisting Data
    vector_db = Chroma.from_documents(
        chunks, embedding=GPT4AllEmbeddings(), persist_directory="vector_cache/"
    )
    vector_db.persist()  # Save embeddings persistently
    return vector_db

# Initialize vector DB if documents exist
vector_db = process_documents()
if vector_db:
    st.session_state.vector_db = vector_db

# Function to query GPT4All with document context
def query_gpt4all_with_docs(user_input):
    vector_db = st.session_state.get("vector_db", None)
    if not vector_db:
        return "⚠️ No documents found. Please add files to `documents/` first."

    search_results = vector_db.similarity_search(user_input, k=1)  # Reduce k for speed
    retrieved_text = "\n".join([doc.page_content for doc in search_results])

    prompt = f"Relevant Information:\n{retrieved_text}\n\nUser: {user_input}\nAssistant:"
    response = local_model.generate(prompt, max_tokens=100)  # Reduce max_tokens for speed

    return response

# Rental Application Form
st.title("Strawberry Oaks Rental Application - AI Assistant")
with st.form("my_form"):
    st.text_input("First and Last Name")
    st.text_input("Current Address")
    st.text_input("Phone Number")
    st.form_submit_button("Submit")

# Sidebar with Example Questions & Chatbot
with st.sidebar:
    st.title("Ask a question about the rental agreement")
    
    st.write("## Example Questions")
    st.write("Want to know where the nearest pizza shop is? Ask it here! 🍕")
    st.write("What is included in the rent (utilities, parking, etc)? 💸")
    st.write("Is there a pet deposit or a monthly fee? 🐕‍🦺")
    st.write("Are there nearby construction projects or busy roads? 🚧")

    # Move Chatbot to Sidebar
    user_question = st.text_input("Type your question here:")

    if user_question:
        answer = query_gpt4all_with_docs(user_question)
        st.write("### Answer:")
        st.write(answer)
