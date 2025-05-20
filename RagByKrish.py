from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

import os 
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("bOTAPIKEY")

#generating embeddings
embeddings= GoogleGenerativeAIEmbeddings(
    model= "models/embedding-001", 
    google_api_key= api_key
    )

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a knowledgeable and professional medical assistant developed by Muhammad Hassan, an AI developer.

Your task is to provide clear, concise, and accurate answers to user queries based solely on the provided context. If the context does not contain enough information to answer the question, respond with:
"Sorry, I don't have enough information to answer that."

--- Context ---
{context}

--- User Query ---
{query}

--- Your Response ---
"""
)

def load_vector_store(document_path, vector_store_path):
    print("loading document")
    if document_path.split(".")[-1]=="pdf":
        loader= PyPDFLoader(document_path)
        document= loader.load()
    else :
        loader=TextLoader(document_path)
        document= loader.load()

    print("splitting document") 
    splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_document= splitter.split_documents(document)

    print("creating vector store")
    vector_store= FAISS.from_documents(chunked_document, embeddings)

    print ("saving vector store locally")
    vector_store.save_local(vector_store_path)
    print("done")

def get_context(retriever, query):
    context= retriever.invoke(query)
    return context 

def get_response(retriever, query):
    context=get_context(retriever, query)
    formatted_prompt=prompt.format(context=context, query=query)
    response=model.invoke(formatted_prompt)
    return response

def create_vector_store(vector_store_path):
    retriever=FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True).as_retriever()
    return retriever