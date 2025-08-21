from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()
api_key=os.getenv("genai_api")

#creating embeddings
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

#Model
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

#prompt
prompt_template=ChatPromptTemplate.from_template('''

you are a helpful fitness trainer build by hassan hamidi.your work is to guide user in their fitness journey 
make sure to solve users query and answer them in a very intelligent and concise way.
If the user ask any question which is not in the given context.just tell them 
"Sorry , I dont have enough information "

context:
{context}

query:
{query}
'''
)

def create_vector_store(document_path, vector_store_path):
    print("loading the data")
    loader=PyPDFLoader(document_path)
    document=loader.load()

    print("apply chunking")
    splitter =RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_document=splitter.split_documents(document)

    print('creating vector store ')
    vector_store= FAISS.from_documents(chunked_document, embeddings)

    print("saving vector store locally")
    vector_store.save_local(vector_store_path)
    print("Done")


def load_vector_store(vector_store_path):
    retriever=FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True).as_retriever()
    return retriever

def get_context(retriever, querry):
    context=retriever.invoke(querry)
    return context      

def get_response(retriever, query):
    context=get_context(retriever, query)
    prompt=prompt_template.format(context=context, query=query)
    response=model.invoke(query)
    return response.content