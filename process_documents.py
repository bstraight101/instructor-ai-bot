import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain.schema import Document

def load_documents(upload_dir):
    docs = []
    for file in os.listdir(upload_dir):
        path = os.path.join(upload_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_documents(documents):
    smart_chunks = []

    for doc in documents:
        # Split on double newlines (usually paragraph or section breaks)
        raw_chunks = doc.page_content.split("\n\n")

        for chunk in raw_chunks:
            chunk = chunk.strip()

            # Ignore empty or tiny chunks
            if len(chunk) < 50:
                continue

            # Group nearby lines (helps with bullet lists, date lines, etc.)
            grouped_chunk = "\n".join(line.strip() for line in chunk.split("\n") if line.strip())

            smart_chunks.append(Document(page_content=grouped_chunk, metadata=doc.metadata))

    return smart_chunks
