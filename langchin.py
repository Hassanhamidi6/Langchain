from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API key
api_key = os.getenv("bOTAPIKEY")

# Streamlit app UI
st.title("Generative AI Chatbot")
input_text = st.text_input("Enter your query:")

# Prompt Template (correct grammar and format)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Make sure to answer every query asked by the user in a very intelligent and concise way."),
    ("user", "Query: {query}")
])

# Google Generative AI LLM with API key
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Output parser
output_parser = StrOutputParser()

# Combine prompt, model, and parser into a chain
chain = prompt | llm | output_parser

# Run when user provides input
if input_text:
    response = chain.invoke({"query": input_text})
    st.write(response)
