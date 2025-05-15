import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_PATH = "vectorstore"

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_PATH)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    prompt_template = """
    You are a helpful instructor AI assistant. Use the following course content to answer the student's question.
    If you do not know the answer based on the course content, say "I don't have that information, please check the syllabus or ask your instructor."
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm = Ollama(model="mistral")  # Uses local model running on Ollama
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
