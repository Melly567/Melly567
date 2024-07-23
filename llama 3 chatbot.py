import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import fitz  # PyMuPDF
import io
from sentence_transformers import SentenceTransformer
import requests

# Function to extract text and page number from PDF using PyMuPDF
@st.cache_data
def pdf_loader(file):
    pdf_bytes = file.read()
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    texts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        texts.append({"page": page_num, "text": text})
    return texts

# Load and extract text from multiple PDFs
def load_pdfs(files):
    pdf_texts = []
    for file in files:
        pdf_texts.extend(pdf_loader(file))
    return pdf_texts

# Function to generate a response using Groq API
def generate_groq_response(prompt, groq_api_key):
    headers = {
        'Authorization': f'Bearer {groq_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'llama3-8b',  # Example model, adjust as necessary
        'prompt': prompt,
        'max_tokens': 500
    }
    response = requests.post('https://api.groq.com/generate', headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Initialize BERT model for embeddings
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('bert-base-nli-mean-tokens')

bert_model = load_bert_model()

# Streamlit interface
st.title("Llama 3 Chatbot using Langchain")

# Custom CSS
st.markdown("""
<style>
body {
    font-family: 'Times New Roman', serif;
}
header {
    background-color: #800080;  /* Magenta/Purple color */
    color: white;
    text-align: center;
    padding: 1em;
    border-radius: 5px;
}
.upload-area {
    border: 2px dashed #800080;  /* Magenta/Purple color */
    border-radius: 10px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<header>Upload PDF Documents and Ask Questions</header>", unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type="pdf")

if uploaded_files:
    st.write("Processing uploaded files...")
    pdf_texts_with_pages = load_pdfs(uploaded_files)
    
    if pdf_texts_with_pages:
        st.success("PDF files uploaded and text extracted successfully.")
        
        # Define chunk size, overlap, and separators
        chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, chunk_size // 2, 100)
        separators = st.sidebar.text_input("Separators", value=",.;\n")

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            separators=list(separators)
        )

        # Split the data into chunks
        st.write("Splitting text into chunks...")
        documents = text_splitter.split_documents(pdf_texts_with_pages)

        # Create embeddings for the chunks using BERT
        st.write("Creating embeddings for the chunks...")
        embeddings = bert_model.encode([doc.page_content for doc in documents])

        st.success("Text split into chunks and embeddings generated successfully.")

        # Initialize FAISS vector store
        st.write("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(documents, embeddings)
        st.success("Vector store created successfully.")

        # Create chatbot interface
        st.header("Ask a Question")
        query = st.text_input("Enter your question based on the uploaded documents")
        groq_api_key = st.text_input("Enter your Groq API key", type="password")

        if query and groq_api_key:
            # Retrieve the most similar documents to the query
            st.write("Retrieving the most relevant documents...")
            retriever = vector_store.as_retriever()
            result = retriever.retrieve(query, top_k=1)
            
            # Display the results
            st.write("Response:")
            context = " ".join([doc.page_content for doc in result])
            for doc in result:
                st.write(f"Page {doc.metadata['page']}: {doc.page_content}")

            # Use Groq API to generate an answer based on the retrieved documents
            prompt = f"Answer the question based on the provided context.\nContext: {context}\nQuestion: {query}"
            answer = generate_groq_response(prompt, groq_api_key)
            st.write("Answer:")
            st.write(answer)
    else:
        st.error("No text extracted from the uploaded PDF files.")
else:
    st.write("No files uploaded yet.")