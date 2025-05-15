import os
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = DocArrayInMemorySearch.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

def load_vectorstore():
    # For DocArrayInMemorySearch, vectorstore is recreated during this session
    return None

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    prompt_template = '''
    You are a helpful instructor AI assistant. Use the following course content to answer the student's question.
    If you do not know the answer based on the course content, say "I don't have that information, please check the syllabus or ask your instructor."

    Context:
    {context}

    Question: {question}
    Answer:
    '''
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
