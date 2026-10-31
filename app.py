import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Screener")
st.title("AI Resume Screener - ATS")

jd = st.text_area("Paste Job Description Here:")

files = st.file_uploader("Upload Resumes in PDF", type="pdf", accept_multiple_files=True)

if st.button("Rank Resumes"):
    if jd and files:
        texts = []
        names = []
        for f in files:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            txt = ""
            for page in doc:
                txt += page.get_text()
            txt = re.sub(r'\s+', ' ', txt)
            texts.append(txt)
            names.append(f.name)
        
        vec = TfidfVectorizer(stop_words='english')
        vectors = vec.fit_transform([jd] + texts)
        scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
        
        for name, score in sorted(zip(names, scores), key=lambda x: x[1], reverse=True):
            per = round(score*100, 2)
            result = "SELECTED" if per > 60 else "REJECTED"
            st.write(f"{name} - {per}% - {result}")
            st.progress(int(per))
    else:
        st.warning("Paste JD and upload PDFs")
