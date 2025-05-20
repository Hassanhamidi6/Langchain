from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()

app=FastAPI(
    title="Langchain Server"  ,  
    version="1.0",
    description= "A simple API Server"
)


#promt template
prompt= ChatPromptTemplate.from_messages([
    ("Write an essay about {essay}  of 100 words ")
])

llm= ChatOllama(model="llama3.2:1b")

add_routes(
    app, 
    prompt|llm, 
    path= "/essay"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)