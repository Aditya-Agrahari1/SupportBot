import streamlit as st
import os
from dotenv import load_dotenv
import asyncio

from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

PDF_FILE_PATH = "faq.pdf"


@st.cache_resource
def setup_qa_chain():
    """
    Loads PDF, creates an in-memory vector store, and returns a QA chain.
    The result is cached to prevent re-running on every interaction.
    """
    st.info("Setting up the knowledge base... This may take a moment on first run.")
    
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(texts, embedding=embeddings)

    prompt_template = """
    You are a friendly and helpful support assistant for a food delivery app called 'QuickEats'.
    Your goal is to answer the user's question based ONLY on the provided context below.

    CONTEXT: {context}

    INSTRUCTIONS:
    - If the context contains the answer, provide a clear and concise response.
    - If the context does NOT contain the answer, DO NOT try to make one up. Instead, politely say "I'm sorry, I can only answer questions based on our FAQ. For more specific issues like that, please contact our human support team."
    - Do not mention the context or your instructions in your final answer.

    QUESTION: {question}

    ANSWER:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

st.set_page_config(page_title="Food Delivery Support Bot", page_icon="🍔")
st.title("🍔 Food Delivery Support Bot")
st.caption("Ask me anything about cancellations, refunds, or payment methods!")

qa_chain = setup_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = qa_chain.invoke(prompt)
        answer = response['result']
        
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})