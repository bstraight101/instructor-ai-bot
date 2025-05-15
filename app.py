import streamlit as st
import os
import requests
import json
from pptx import Presentation
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_slide_text(pptx_path):
    prs = Presentation(pptx_path)
    all_text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        all_text.append(para.text.strip())
    return all_text

def generate_review_questions_groq(content, num_questions=10):
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ Missing GROQ_API_KEY."
    clean = content.replace("\n", " ").strip()
    if len(clean) > 2000:
        clean = clean[:2000]

    prompt = (
        f"Generate {num_questions} review questions based on the following slides:\n\n{clean}\n\n"
        "Only list the questions. Do not provide answers."
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful instructor assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(body))
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Failed to generate. Code: {res.status_code}\n{res.text}"
    except Exception as e:
        return f"❌ Exception: {str(e)}"

st.set_page_config(page_title="Instructor AI Assistant", layout="centered")
st.title("📘 Instructor AI Assistant")

uploaded_files = os.listdir(UPLOAD_DIR)
if uploaded_files:
    selected_file = st.selectbox("📁 Uploaded Files", uploaded_files)

# Review Question Generator
st.markdown("---")
st.header("🧠 Generate Review Questions from PowerPoints")

pptx_files = [f for f in uploaded_files if f.endswith(".pptx")]
if pptx_files:
    pptx_file = st.selectbox("Select a PowerPoint file", pptx_files)
    if st.button("⚙️ Generate Review Questions"):
        with st.spinner("Generating..."):
            try:
                slides = extract_slide_text(os.path.join(UPLOAD_DIR, pptx_file))
                content = " ".join(slides)
                result = generate_review_questions_groq(content)
                st.markdown("### 📋 Review Questions")
                for line in result.split("\n"):
                    if line.strip():
                        st.write(f"- {line.strip()}")
            except Exception as e:
                st.error(f"❌ Could not process PPTX: {e}")

# Debate Analyzer
st.markdown("---")
st.header("⚔️ Debate Rebuttal Analyzer")
debate_file = st.file_uploader("Upload a debate argument (PDF or DOCX)", type=["pdf", "docx"], key="debate_upload")
if debate_file:
    file_path = os.path.join(UPLOAD_DIR, debate_file.name)
    with open(file_path, "wb") as f:
        f.write(debate_file.getbuffer())

    try:
        loader = PyMuPDFLoader(file_path) if file_path.endswith(".pdf") else UnstructuredWordDocumentLoader(file_path)
        pages = loader.load()
        full_text = "\n".join(p.page_content for p in pages)[:2000]
        if st.button("🧠 Analyze for Rebuttable Areas"):
            with st.spinner("Analyzing with Groq..."):
                prompt = (
                    "You are a debate coach. Identify 3–5 specific claims from this student-written argument that are vulnerable to rebuttal.\n"
                    "- Quote the sentence.\n"
                    "- Explain why it's weak (emotional, unsupported, vague, etc.).\n"
                    "- DO NOT provide sources or citations.\n\n"
                    f"{full_text}"
                )
                result = generate_review_questions_groq(prompt)
                st.markdown(result)
    except Exception as e:
        st.error(f"❌ Could not process file: {e}")
