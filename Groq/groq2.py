import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import retrieval_qa
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time 

load_dotenv()

#loading groq and genai key

groq_api_key=os.getenv("groq_api_key")
genai_api_key=os.getenv("genai_api")

st.title("Chat Groq with Llama")

llm=ChatGroq(model="llama-3.3-70b-versatile ", api_key=groq_api_key)
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant built by Hassan Hamidi. "
               "Your job is to assist users with the best context in a very intelligent, concise, and easy-to-understand way. "
               "Make sure the user is satisfied with your answer."),
    ("human", "<context>\n{context}\n\nQuestion: {input}")
])

def vector_embeddings():
        
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./PDFs") #Data Ingestion
        st.session_state.docs=st.session_state.loader.load()  # Loading Document

        st.session_state.splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #chunk creation 
        st.session_state.chunked_document= st.session_state.splitter.split_documents(st.session_state.docs[:20]) #Split

        st.session_state.vectors=FAISS.from_documents(st.session_state.chunked_document, st.session_state.embeddings)

prompt1=st.text_input("Enter your question from the docs")


if st.button("Document Embeddings"):
    vector_embeddings()
    st.write("Vector store db is ready")

 
if prompt1:
    start=time.process_time()
    documnet_chain=create_stuff_documents_chain(llm, prompt)
    retriever= st.session_state.vectors.as_retriever()
    retrieval_chian=retrieval_qa(retriever, documnet_chain)

    response= retrieval_chian.invoke({"input":prompt1})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])

    #With a ateamlit expader
    with st.expander("Document Similarity Search"): 
        relevant_docs = retriever.get_relevant_documents(prompt1)
    for i, doc in enumerate(relevant_docs):
        st.write(doc.page_content)
        st.write("-------------------------------")
    