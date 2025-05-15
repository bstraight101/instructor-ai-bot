import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIR = "vectorstore"

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    llm = ChatOllama(model="mistral")
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
