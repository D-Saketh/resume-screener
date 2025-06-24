
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import os
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screening Tool", layout="centered")

st.title("📄 AI-Powered Resume Screening Tool")

st.markdown("""
Upload a **job description (PDF)** and **multiple resumes (PDFs)**.  
The app will extract text, preprocess it using spaCy, compare each resume to the job description using TF-IDF + cosine similarity, and rank them accordingly.
""")

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Clean and preprocess text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return " ".join([page.get_text() for page in doc])

job_file = st.file_uploader("Upload Job Description (PDF)", type="pdf", key="job")
resume_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True, key="resumes")

if job_file and resume_files:
    st.success("Files uploaded. Processing...")

    job_text = clean_text(extract_text_from_pdf(job_file))
    resumes_text = {res.name: clean_text(extract_text_from_pdf(res)) for res in resume_files}

    documents = [job_text] + list(resumes_text.values())
    resume_names = list(resumes_text.keys())

    # Vectorize and compute similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    results = pd.DataFrame({
        "Resume Name": resume_names,
        "Similarity Score": scores
    }).sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)

    st.subheader("📊 Ranked Resumes")
    st.dataframe(results)

    # Download CSV
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "ranked_resumes.csv", "text/csv")
