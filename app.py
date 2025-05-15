import streamlit as st
import os
import shutil
import csv
import difflib
import requests
import pandas as pd
from pptx import Presentation
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from process_documents import load_documents, split_documents
from qa_chain import create_vectorstore, load_vectorstore, build_qa_chain

# === CONFIG ===
UPLOAD_DIR = "uploads"
VECTOR_PATH = "vectorstore"
CUSTOM_QA_FILE = "custom_qa.csv"
BLOCKED_QA_FILE = "blocked_quiz_questions.csv"
GENERATED_DIR = "generated_questions"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# === Load Q&A ===
def load_custom_qna(csv_path=CUSTOM_QA_FILE):
    qna_dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    qna_dict[row[0].strip().lower()] = row[1].strip()
    return qna_dict

def find_best_match(user_input, qna_dict, threshold=0.85):
    user_input = user_input.strip().lower()
    best_match = difflib.get_close_matches(user_input, qna_dict.keys(), n=1, cutoff=threshold)
    return qna_dict[best_match[0]] if best_match else None

custom_qna = load_custom_qna()

def load_blocked_questions(csv_path=BLOCKED_QA_FILE):
    blocklist = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    blocklist.add(row[0].strip().lower())
    return blocklist

def is_blocked_question(user_input, blocked_set, threshold=0.85):
    user_input = user_input.strip().lower()
    return bool(difflib.get_close_matches(user_input, blocked_set, n=1, cutoff=threshold))

blocked_questions = load_blocked_questions()

# === STREAMLIT SETUP ===
st.set_page_config(page_title="Instructor AI Assistant", layout="centered")
st.title("📘 Instructor AI Assistant")

if st.button("🔁 Reset All"):
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(VECTOR_PATH, ignore_errors=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.experimental_rerun()

if os.listdir(UPLOAD_DIR):
    st.subheader("📁 Uploaded Files")
    selected_file = st.selectbox(
        "Currently loaded into the AI assistant:",
        sorted(os.listdir(UPLOAD_DIR)),
        key="uploaded_file_dropdown"
    )

vectorstore, qa = None, None
if os.path.exists(os.path.join(VECTOR_PATH, "index.faiss")):
    with st.spinner("🔁 Loading saved memory..."):
        vectorstore = load_vectorstore()
        qa = build_qa_chain(vectorstore)
elif os.listdir(UPLOAD_DIR):
    with st.spinner("📚 Processing files..."):
        docs = load_documents(UPLOAD_DIR)
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
        qa = build_qa_chain(vectorstore)
        st.success("✅ Assistant Ready")

if qa:
    query = st.text_input("What do you want to know?")
    if query:
        with st.spinner("🤖 Thinking..."):
            if is_blocked_question(query, blocked_questions):
                st.warning("❌ I'm not allowed to answer quiz/exam questions.")
            else:
                answer = find_best_match(query, custom_qna)
                st.write(answer if answer else qa.run(query))
