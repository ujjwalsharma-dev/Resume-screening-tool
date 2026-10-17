"""
Resume Screening Tool
Ranks uploaded resumes (PDF) against a job description using
TF-IDF vectorization and cosine similarity.
"""

import re

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Screening Tool", page_icon="📄", layout="wide")


# ---------- Helper functions ----------
def extract_text(uploaded_file) -> str:
    """Read all pages of a PDF and return the combined text."""
    try:
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/emails/special characters, collapse spaces."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rank_resumes(job_description: str, resumes: dict, top_n_keywords: int = 25):
    """
    resumes: {filename: cleaned_text}
    Returns a DataFrame with match scores and keyword analysis.
    """
    names = list(resumes.keys())
    docs = [clean_text(job_description)] + [resumes[n] for n in names]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)

    # Cosine similarity between the job description (row 0) and each resume
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    # Top keywords from the job description, using unigram TF-IDF weights
    kw_vectorizer = TfidfVectorizer(stop_words="english")
    kw_matrix = kw_vectorizer.fit_transform(docs)
    terms = kw_vectorizer.get_feature_names_out()
    jd_weights = kw_matrix[0].toarray().flatten()
    top_idx = jd_weights.argsort()[::-1][:top_n_keywords]
    jd_keywords = [terms[i] for i in top_idx if jd_weights[i] > 0]

    rows = []
    for name, score in zip(names, scores):
        resume_words = set(resumes[name].split())
        matched = [k for k in jd_keywords if k in resume_words]
        missing = [k for k in jd_keywords if k not in resume_words]
        coverage = len(matched) / len(jd_keywords) * 100 if jd_keywords else 0
        rows.append(
            {
                "Resume": name,
                "Match Score (%)": round(score * 100, 1),
                "Keyword Coverage (%)": round(coverage, 1),
                "Matched Keywords": ", ".join(matched),
                "Missing Keywords": ", ".join(missing),
            }
        )

    df = pd.DataFrame(rows).sort_values("Match Score (%)", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True), jd_keywords


# ---------- UI ----------
st.title("📄 Resume Screening Tool")
st.caption("Rank resumes against a job description using TF-IDF and cosine similarity.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="e.g. We are looking for a Python developer with experience in "
        "Django, REST APIs, SQL, Git and problem-solving skills...",
    )

with col2:
    st.subheader("2. Upload Resumes")
    uploaded = st.file_uploader(
        "Upload one or more resumes (PDF)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded:
        st.success(f"{len(uploaded)} resume(s) uploaded")

if st.button("Screen Resumes", type="primary"):
    if not job_description.strip():
        st.warning("Please paste a job description.")
    elif not uploaded:
        st.warning("Please upload at least one PDF resume.")
    else:
        resumes, unreadable = {}, []
        with st.spinner("Reading and analysing resumes..."):
            for f in uploaded:
                text = clean_text(extract_text(f))
                if text:
                    resumes[f.name] = text
                else:
                    unreadable.append(f.name)

        if unreadable:
            st.warning(
                "Could not read text from: "
                + ", ".join(unreadable)
                + " (scanned/image PDFs are not supported)."
            )

        if resumes:
            df, jd_keywords = rank_resumes(job_description, resumes)

            st.subheader("Results")
            best = df.iloc[0]
            st.metric("Top candidate", best["Resume"], f"{best['Match Score (%)']}% match")

            st.dataframe(
                df[["Rank", "Resume", "Match Score (%)", "Keyword Coverage (%)"]],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Score comparison")
            st.bar_chart(df.set_index("Resume")["Match Score (%)"])

            st.subheader("Keyword analysis")
            st.caption("Top keywords: " + ", ".join(jd_keywords))
            for _, row in df.iterrows():
                with st.expander(f"#{row['Rank']}  {row['Resume']}  ({row['Match Score (%)']}%)"):
                    st.markdown(f"**✅ Matched:** {row['Matched Keywords'] or 'None'}")
                    st.markdown(f"**❌ Missing:** {row['Missing Keywords'] or 'None'}")

            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="resume_screening_results.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Limitation: this tool matches keywords statistically and does not understand "
    "meaning. Future work: sentence embeddings (BERT), skill extraction, OCR for scanned PDFs."
)
