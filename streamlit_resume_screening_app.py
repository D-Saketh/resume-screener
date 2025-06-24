import streamlit as st
import fitz
import os
import pandas as pd
import spacy
import re
import subprocess
import importlib.util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI-Powered Resume Screening Tool")

# 🧠 Load spaCy model with fallback
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    if importlib.util.find_spec(model_name) is None:
        subprocess.run(["python", "-m", "spacy", "download", model_name])
    return spacy.load(model_name)

nlp = load_spacy_model()

# 📁 Upload section
uploaded_job = st.file_uploader("Upload Job Description (PDF)", type="pdf", key="job")
uploaded_resumes = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True, key="resumes")

if uploaded_job and uploaded_resumes:
    with st.spinner("Processing..."):

        os.makedirs("temp", exist_ok=True)
        job_path = os.path.join("temp", "job.pdf")
        with open(job_path, "wb") as f:
            f.write(uploaded_job.read())

        resume_paths = []
        for uploaded in uploaded_resumes:
            path = os.path.join("temp", uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.read())
            resume_paths.append(path)

        # 📄 Extract text from PDF
        def extract_text_from_pdf(file_path):
            doc = fitz.open(file_path)
            return " ".join([page.get_text() for page in doc])

        job_text = extract_text_from_pdf(job_path)
        resumes_text = {os.path.basename(path): extract_text_from_pdf(path) for path in resume_paths}

        # 🧹 Text cleaning
        def clean_text(text):
            text = re.sub(r'[^a-zA-Z ]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()

        job_text_clean = clean_text(job_text)
        resumes_text_clean = {name: clean_text(text) for name, text in resumes_text.items()}

        # 📊 TF-IDF + cosine similarity
        documents = [job_text_clean] + list(resumes_text_clean.values())
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # 📈 Ranking
        ranking_df = pd.DataFrame({
            "Resume Name": list(resumes_text_clean.keys()),
            "Similarity Score": similarity_scores
        }).sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)

        st.success("✅ Ranking Complete!")
        st.dataframe(ranking_df)

        # 💾 Export CSV
        csv = ranking_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "ranked_resumes.csv", "text/csv")
