import streamlit as st
import os
import requests
import json
import csv
import difflib
import pandas as pd
from pptx import Presentation
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from process_documents import load_documents, split_documents
from qa_chain import create_vectorstore, build_qa_chain

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

def generate_with_groq(prompt):
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ Missing GROQ_API_KEY."
    clean = prompt.replace("\n", " ").strip()
    if len(clean) > 2000:
        clean = clean[:2000]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful instructor assistant."},
            {"role": "user", "content": clean}
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

def generate_review_questions(slide_texts, num_questions=10):
    joined = " ".join(slide_texts)
    return generate_with_groq(
        f"Generate {num_questions} open-ended review questions based on the following content.\n\n{joined}\n\nOnly list the questions. No answers."
    )

# === Streamlit UI ===
st.set_page_config(page_title="Instructor AI Assistant", layout="centered")
st.title("📘 Instructor AI Assistant")

if st.button("🔁 Reset All"):
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))
    st.experimental_rerun()

uploaded_files = os.listdir(UPLOAD_DIR)
if uploaded_files:
    st.selectbox("📁 Uploaded Files", uploaded_files)

# === Vectorstore memory setup ===
qa = None
if uploaded_files:
    with st.spinner("📚 Processing documents..."):
        docs = load_documents(UPLOAD_DIR)
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
        qa = build_qa_chain(vectorstore)

# === QA Section ===
custom_qna = load_custom_qna()
blocked_questions = load_blocked_questions()

st.markdown("---")
st.header("💬 Ask a Question")

if qa:
    query = st.text_input("What do you want to know?")
    if query:
        with st.spinner("🤖 Thinking..."):
            if any(difflib.get_close_matches(query.strip().lower(), blocked_questions, n=1, cutoff=0.85)):
                st.warning("❌ I'm not allowed to answer quiz/exam questions.")
            else:
                answer = find_best_match(query, custom_qna)
                st.write(answer if answer else qa.run(query))

# === Review Question Generator ===
st.markdown("---")
st.header("🧠 Generate Review Questions from PowerPoints")
pptx_files = [f for f in uploaded_files if f.endswith(".pptx")]
if pptx_files:
    pptx_file = st.selectbox("Select a PowerPoint file", pptx_files)
    if st.button("⚙️ Generate Review Questions"):
        with st.spinner("Generating..."):
            try:
                slides = extract_slide_text(os.path.join(UPLOAD_DIR, pptx_file))
                result = generate_review_questions(slides)
                st.markdown("### 📋 Review Questions")
                for line in result.split("\n"):
                    if line.strip():
                        st.write(f"- {line.strip()}")
            except Exception as e:
                st.error(f"❌ Could not process PPTX: {e}")

# === Quiz Preview Mode ===
st.markdown("---")
st.header("📝 Quiz Preview Mode")
quiz_files = [f for f in uploaded_files if f.endswith("_review.csv")]
if quiz_files:
    selected_quiz = st.selectbox("Choose a quiz file", quiz_files)
    try:
        df = pd.read_csv(os.path.join(UPLOAD_DIR, selected_quiz))
        required_cols = ['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Option Index']
        if all(col in df.columns for col in required_cols):
            show_answers = st.checkbox("✅ Show Correct Answers")
            clean_df = df.dropna(subset=required_cols)
            for i, row in clean_df.iterrows():
                st.markdown(f"**Q{i+1}: {row['Question']}**")
                st.write(f"A) {row['Option A']}")
                st.write(f"B) {row['Option B']}")
                st.write(f"C) {row['Option C']}")
                st.write(f"D) {row['Option D']}")
                if show_answers:
                    try:
                        correct_letter = ['A', 'B', 'C', 'D'][int(row['Correct Option Index'])]
                        st.success(f"✔️ Correct Answer: {correct_letter}")
                    except Exception:
                        st.warning("⚠️ Invalid answer index.")
                st.markdown("---")
    except Exception as e:
        st.error(f"❌ Error reading quiz file: {e}")

# === Debate Rebuttal Analyzer ===
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
                result = generate_with_groq(prompt)
                disclaimer = (
                    "\n\n---\n\n**Note:** You are required to find academic sources to support or refute the claims above. "
                    "Do not follow them blindly. These suggestions are for brainstorming, not final arguments."
                )
                st.markdown(result + disclaimer)
    except Exception as e:
        st.error(f"❌ Could not process file: {e}")
