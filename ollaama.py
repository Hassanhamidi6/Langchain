import streamlit as st
from langchain_community.chat_models import ChatOllama  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("langchain bot ")

question = st.text_input("Ask a question:")

if question:
    llm = ChatOllama(model="llama3.2:1b")  
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. built by Muhammad Hassan an Artificial Intelligence developer"),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"question": question})

    st.write("### Response:")
    st.write(response)
