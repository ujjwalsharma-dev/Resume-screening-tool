# 📄 Resume Screening Tool

A web app that ranks resumes against a job description using **TF-IDF** and **cosine similarity**.

**Live demo:** <add your Streamlit Cloud link here>

## Features
- Paste a job description and upload multiple PDF resumes
- Ranks candidates by match score (%)
- Keyword coverage: shows matched and missing keywords for each resume
- Bar chart comparison of candidates
- Download results as CSV

## Tech Stack
Python, Streamlit, scikit-learn, pypdf, pandas

## How it works
1. Extract text from each PDF using `pypdf`
2. Clean the text (lowercase, remove URLs, emails, special characters)
3. Convert the job description and resumes into TF-IDF vectors (unigrams + bigrams, English stopwords removed)
4. Compute cosine similarity between the job description and each resume
5. Pick the top 25 job-description keywords and check which ones each resume contains

## Run locally
```bash
git clone <your-repo-url>
cd resume-screener
pip install -r requirements.txt
streamlit run app.py
```

## Limitations and future work
- Keyword-based matching, so it does not understand meaning (e.g. "ML" vs "machine learning")
- Scanned/image PDFs are not supported (could add OCR with Tesseract)
- Future: sentence embeddings (BERT), skill extraction with NER, DOCX support

## Screenshots
<add screenshots here>
