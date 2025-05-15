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
from qa_chain import create_vectorstore, build_qa_chain
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader

UPLOAD_DIR = "uploads"
CUSTOM_QA_FILE = "custom_qa.csv"
BLOCKED_QA_FILE = "blocked_quiz_questions.csv"

os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_custom_qna(csv_path=CUSTOM_QA_FILE):
    qna = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    qna[row[0].strip().lower()] = row[1].strip()
    return qna

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

def find_best_match(user_input, qna_dict, threshold=0.85):
    user_input = user_input.strip().lower()
    best_match = difflib.get_close_matches(user_input, qna_dict.keys(), n=1, cutoff=threshold)
    return qna_dict[best_match[0]] if best_match else None

def is_blocked_question(user_input, blocked_set, threshold=0.85):
    user_input = user_input.strip().lower()
    return bool(difflib.get_close_matches(user_input, blocked_set, n=1, cutoff=threshold))

def extract_slide_text(pptx_path):
    prs = Presentation(pptx_path)
    return [
        " ".join(para.text.strip() for para in shape.text_frame.paragraphs if para.text.strip())
        for slide in prs.slides for shape in slide.shapes if shape.has_text_frame
    ]

def generate_with_groq(prompt):
    api_key = st.secrets["GROQ_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Failed to generate. Status code: {res.status_code}"

def generate_review_questions(slide_texts, num_questions=10):
    prompt = (
        f"Generate {num_questions} open-ended review questions for students based on this PowerPoint content.\n\n"
        + "\n".join(slide_texts[:15]) +
        "\n\nOnly list the questions. No answers."
    )
    return generate_with_groq(prompt)

def generate_rebuttal_analysis(text):
    prompt = (
        "You are a debate coach. Identify 3–5 specific claims from this student-written argument that are vulnerable to rebuttal.\n"
        "- Quote the sentence.\n"
        "- Explain why it's weak (emotional, unsupported, vague, etc.).\n"
        "- DO NOT provide sources or citations.\n\n"
        f"{text}\n\n"
        "Respond as a list with quoted text and a critique."
    )
    return generate_with_groq(prompt)

st.set_page_config(page_title="Instructor AI Assistant", layout="centered")
st.title("📘 Instructor AI Assistant")

if st.button("🔁 Reset All"):
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    st.experimental_rerun()

if os.listdir(UPLOAD_DIR):
    st.subheader("📁 Uploaded Files")
    st.selectbox("Currently loaded into the AI assistant:", sorted(os.listdir(UPLOAD_DIR)))

qa = None
if os.listdir(UPLOAD_DIR):
    with st.spinner("📚 Processing files..."):
        docs = load_documents(UPLOAD_DIR)
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
        qa = build_qa_chain(vectorstore)
        st.success("✅ Assistant Ready")

custom_qna = load_custom_qna()
blocked_questions = load_blocked_questions()

if qa:
    query = st.text_input("What do you want to know?")
    if query:
        with st.spinner("🤖 Thinking..."):
            if is_blocked_question(query, blocked_questions):
                st.warning("❌ I'm not allowed to answer quiz/exam questions.")
            else:
                answer = find_best_match(query, custom_qna)
                st.write(answer if answer else qa.run(query))

st.markdown("---")
st.header("🧠 Generate Review Questions from PowerPoints")
pptx_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pptx")]
if pptx_files:
    selected_pptx = st.selectbox("Select a PowerPoint file", pptx_files)
    if st.button("⚙️ Generate Review Questions"):
        with st.spinner("⏳ Generating..."):
            slides = extract_slide_text(os.path.join(UPLOAD_DIR, selected_pptx))
            result = generate_review_questions(slides)
            st.markdown("### 📋 Review Questions")
            for line in result.split("\n"):
                if line.strip():
                    st.write(f"- {line.strip()}")

st.markdown("---")
st.header("⚔️ Debate Rebuttal Analyzer")

debate_file = st.file_uploader("Upload a debate argument (PDF or DOCX)", type=["pdf", "docx"], key="debate_upload")
if debate_file:
    file_path = os.path.join(UPLOAD_DIR, debate_file.name)
    with st.spinner("📤 Uploading file..."):
        with open(file_path, "wb") as f:
            f.write(debate_file.getbuffer())

    try:
        with st.spinner("📄 Extracting text..."):
            loader = UnstructuredPDFLoader(file_path) if file_path.endswith(".pdf") else UnstructuredWordDocumentLoader(file_path)
            pages = loader.load()
            full_text = "\n".join([p.page_content for p in pages])
    except Exception as e:
        st.error(f"❌ Could not process file: {e}")
        full_text = None

    if full_text and st.button("🧠 Analyze for Rebuttable Areas"):
        with st.spinner("Analyzing with Groq..."):
            feedback = generate_rebuttal_analysis(full_text)
            disclaimer = (
                "\n\n---\n\n"
                "**Note:** You are required to find academic sources to support or refute the claims highlighted above. "
                "Some of these claims may be inaccurate or oversimplified. Do not follow them blindly. This analysis is intended to generate ideas, "
                "not provide definitive answers. You are still responsible for conducting your own research to verify the evidence."
            )
            full_output = feedback + disclaimer
            st.markdown(full_output)

            # PDF output
            buffer = BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            text = pdf.beginText(40, 750)
            text.setFont("Helvetica", 11)
            for line in full_output.split("\n"):
                if text.getY() < 40:
                    pdf.drawText(text)
                    pdf.showPage()
                    text = pdf.beginText(40, 750)
                    text.setFont("Helvetica", 11)
                text.textLine(line.strip())
            pdf.drawText(text)
            pdf.save()
            buffer.seek(0)
            st.download_button("📥 Download Feedback as PDF", buffer, "debate_rebuttal_feedback.pdf", mime="application/pdf")
