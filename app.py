from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Prompt template (fixed comma, grammar, and casing)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Make sure to answer every query asked by the user."),
        ("user", "Question: {questions}")
    ]
)

# Streamlit UI
st.title("LangChain Demo with OpenAI")
input_text = st.text_input("Search the topic you want:")

# OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Output parser
output_parser = StrOutputParser()

# Combine the chain
chain = prompt | llm | output_parser

# Function to invoke the chain
def generate_response(query):
    response = chain.invoke({"questions": query})  # Correct key name
    st.write("### Response:")
    st.write(response)

# Trigger the response when there's input
if input_text:
    generate_response(input_text)
