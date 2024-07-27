import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# Define the prompt template for QA
template = """
You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer. If you don't know the context, then don't answer the question.

context: \n{context}\n

question: \n{question}\n
Answer:"""

def get_conversational_chain(retriever):
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.5)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)

    # Define the QA chain directly
    combine_docs_chain = LLMChain(
        llm=model,
        prompt=prompt
    )
    
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain  # Ensure this parameter is correctly named
    )
    return chain

def user_input(user_question):
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(retriever=new_db)  # Pass the valid retriever object
    response = chain({"input_documents": docs, "question": user_question})

    st.write("Answer: ", response.get("output_text", "No answer found."))

# Streamlit UI
st.set_page_config(
    page_title="PDF Analyzer",
    page_icon=':books:',
    layout="wide",
    initial_sidebar_state="auto"
)

with st.sidebar:
    st.title("Upload PDF")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Analyze"):
        if pdf_docs:
            with st.spinner("Analyzing PDF"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.success("VectorDB Uploading Successful!!")
        else:
            st.error("Please upload at least one PDF file.")

def main():
    st.title("LLM GenAi ChatBot")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("🖇 Chat with PDF Analyzer 🗞")
    st.markdown("<hr>", unsafe_allow_html=True)
    user_question = st.text_input("", placeholder="Ask a question about the PDF content")
    st.markdown("<hr>", unsafe_allow_html=True)

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
