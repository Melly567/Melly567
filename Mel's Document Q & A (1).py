#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
import fitz  # PyMuPDF
import os

# Function to extract text and page number from PDF
def extract_text_from_pdf_with_pages(pdf_file):
    doc = fitz.open(pdf_file)
    texts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        texts.append({"page": page_num, "text": text})
    return texts

# Streamlit interface
st.title("Mel's Document Q&A")

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
    pdf_texts_with_pages = []
    for file in uploaded_files:
        pdf_texts_with_pages.extend(extract_text_from_pdf_with_pages(file))
    
    st.success("PDF files uploaded and text extracted successfully.")
    
    # Create DataFrame from extracted text with pages
    df = pd.DataFrame(pdf_texts_with_pages)
    
    # Load the DataFrame into a DataFrameLoader
    df_loader = DataFrameLoader(df, page_content_column="text", metadata_columns=["page"])

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
    documents = df_loader.load_and_split(text_splitter)

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create embeddings for the chunks
    embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])

    st.success("Text split into chunks and embeddings generated successfully.")

    # Initialize FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    st.success("Vector store created successfully.")

    # Create chatbot interface
    st.header("Ask a Question")
    query = st.text_input("Enter your question based on the uploaded documents")

    if query:
        # Retrieve the most similar documents to the query
        retriever = vector_store.as_retriever()
        result = retriever.retrieve(query, top_k=1)
        
        # Display the results
        st.write("Response:")
        for doc in result:
            st.write(f"Page {doc.metadata['page']}: {doc.page_content}")


# In[ ]:




