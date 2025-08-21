from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# API Key
api_key= "AIzaSyD8lcET3a8LNco0pe31Myz1GoARJCkZSFw"

# LLM
model= ChatGoogleGenerativeAI(model= "gemini-2.0-flash", api_key=api_key)

# Prompt
prompt=PromptTemplate(
    input_variables=["query", "context"],
    template='''
    You are a professional medical bot. 
    Use the provided context to answer the user's query.
    If the answer is not in the context, reply:
    "Sorry! I don’t have enough information."
    
    Context:
    {context}
    
    Question:
    {query}
    '''
)

# Load PDF
loader= PyPDFLoader("NIPS-2017-attention-is-all-you-need-Paper.pdf")
documents= loader.load()

# Split into chunks
splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents= splitter.split_documents(documents)

# Embeddings
embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Vector Store
vector_store= FAISS.from_documents(chunked_documents, embeddings)
vector_store.save_local("PDF")

# Retriever
retriever= FAISS.load_local("PDF", embeddings, allow_dangerous_deserialization=True).as_retriever()

# Helper functions
def get_context(retriever, query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def get_response(retriever, query):
    context = get_context(retriever, query)
    formatted_prompt = prompt.format(query=query, context=context)
    response = model.invoke(formatted_prompt)
    return response.content



print("--------------------------------------------------------------------------")


print(get_response(retriever, "What is deep learning ?"))
