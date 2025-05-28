from langchain_groq import ChatGroq 
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from  langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
import streamlit as st
import time
from dotenv import load_dotenv
import os

load_dotenv()
#load groq api key
api_key=os.getenv("groqAPIkey")

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://python.langchain.com/docs/concepts/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.chunked_document=st.session_state.splitter.split_documents(st.session_state.docs)

    st.session_state.vectors=FAISS.from_documents(st.session_state.chunked_document, st.session_state.embeddings)

st.title("Chat Groq")

llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key)

prompt=ChatPromptTemplate.from_messages(
    """Answer the question based on the provided context only
    please provide the most accurate response based on the provided query
    context:
    {context}
    Question:
    {query} 
    """
)

document_chain=create_stuff_documents_chain(llm, prompt)
retriever=st.session_state.vectors.as_retriever()

retrieval_chain=create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Ask any question ")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Reponse time :", time.process_time()-start)
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------")