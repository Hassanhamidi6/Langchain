from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()
api_key=os.getenv("")

model=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=api_key
    )

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

prompt=PromptTemplate(
    input_variables=["context", "query"],
    template='''
you are a very helpful and knowledgeable medical assistant, your work is yo guide users according to their 
query. 
Use the context which is given to you if user ask any query which is not similar to the context just tell them that you dont have enough information

context
{context}

query
{query}
'''
)

def load_vector_store(document_path, vector_store_path):
    if document_path.split(".")[-1]=='pdf':
        loader=PyPDFLoader("")
        document=loader.load()
    else:
        document=WebBaseLoader("").load()


    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_document=splitter.split_documents(document)

    vector_store=FAISS.from_documents(chunked_document, embeddings)

    vector_store.save_local(vector_store_path)

def create_vector_store(vector_store_path):
    retriever=FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True).as_retriever
    return retriever

def get_context(retriever, query):
    context=retriever.invoke(query)
    return context

def get_response(retriever, query):
    context=get_context(retriever, query)
    formated_promp=prompt.format(context=context, query=query)
    response=model.invoke(formated_promp)
    return response
