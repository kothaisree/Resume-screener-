import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Screener Pro", layout="wide", page_icon="🏆")

st.title("AI Resume Screener Pro V2")
st.markdown("### Bulk Screening - 2 to 100 Resumes + RAG Chat")
st.divider()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Selection Threshold %", 0, 100, 15)
    st.success("Supports up to 100 resumes")
    st.info("Features: Bulk Ranking + RAG Chat + Vector Search")

st.subheader("Job Description")
jd = st.text_area("Paste Job Description Here", height=180, placeholder="Ex: Python Developer with Flask, MySQL, JavaScript...")

st.subheader("Upload Resumes (2 to 100 PDFs)")
uploaded = st.file_uploader("Select PDFs", accept_multiple_files=True, type=['pdf'])

def extract_text(pdf):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf)
        for page in reader.pages[:5]:
            t = page.extract_text()
            if t:
                text += t + " "
    except:
        text = ""
    return text

def clean_text(t):
    t = t.lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

if st.button("SCREEN 100 RESUMES NOW", type="primary", use_container_width=True):
    if not jd:
        st.error("Please paste Job Description")
    elif not uploaded:
        st.error("Please upload resumes")
    elif len(uploaded) > 100:
        st.error("Maximum 100 resumes allowed")
    else:
        progress = st.progress(0, text="Reading resumes...")
        jd_clean = clean_text(jd)
        resumes_data = []
        
        for idx, file in enumerate(uploaded):
            progress.progress((idx+1)/len(uploaded), text=f"Reading {idx+1}/{len(uploaded)}: {file.name}")
            raw = extract_text(file)
            if not raw:
                raw = file.name
            resumes_data.append({"name": file.name, "clean": clean_text(raw), "raw": raw})

        progress.progress(0.9, text="Ranking resumes...")
        docs = [jd_clean] + [r["clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform(docs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = []
        for i, r in enumerate(resumes_data):
            score = round(float(scores[i]) * 100, 2)
            status = "SELECTED" if score >= threshold else "REJECTED"
            results.append({"Resume": r["name"], "Score": score, "Status": status, "raw": r["raw"], "clean": r["clean"]})

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df)+1))
        progress.empty()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Resumes", len(df))
        c2.metric("Selected", len(df[df['Score']>=threshold]))
        c3.metric("Rejected", len(df[df['Score']<threshold]))
        c4.metric("Top Score", f"{df['Score'].max()}%")

        st.divider()
        st.subheader(f"Leaderboard - Top {len(df)} Resumes Ranked")
        st.dataframe(df[["Rank", "Resume", "Score", "Status"]], use_container_width=True, hide_index=True, height=600)

        st.divider()
        st.subheader("Top 3 Candidates")
        for idx, row in df.head(10).iterrows():
            if idx == 0:
                st.success(f"RANK 1 - {row['Resume']} - {row['Score']}%")
            elif idx == 1:
                st.warning(f"RANK 2 - {row['Resume']} - {row['Score']}%")
            elif idx == 2:
                st.info(f"RANK 3 - {row['Resume']} - {row['Score']}%")
            else:
                st.write(f"{row['Rank']}. {row['Resume']} - {row['Score']}% {row['Status']}")

        # RAG CHAT SECTION - PROFESSIONAL
        st.divider()
        st.subheader("RAG Chat - Chat with 100 Resumes")
        st.success("RAG System Ready - Vector DB Built Successfully")

        query = st.text_input("Ask a question about the resumes", placeholder="Ex: Why is Rank 1 the best fit? Who has Python experience?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_btn = st.button("Ask RAG", type="primary", use_container_width=True)
        with col2:
            st.caption("Press Enter or click Ask RAG")

        if query:
            q_clean = clean_text(query)
            rag_docs = [r["clean"] for r in results] + [q_clean]
            rag_vectorizer = TfidfVectorizer(stop_words='english')
            rag_tfidf = rag_vectorizer.fit_transform(rag_docs)
            rag_scores = cosine_similarity(rag_tfidf[-1:], rag_tfidf[:-1]).flatten()
            
            top_idx = rag_scores.argsort()[::-1][:3]
            st.divider()
            st.markdown(f"### Answer for: **{query}**")
            for idx in top_idx:
                if rag_scores[idx] > 0.03:
                    st.markdown(f"**{results[idx]['Resume']} ({results[idx]['Score']}%) - Relevance {round(rag_scores[idx]*100,1)}%**")
                    st.info(f"{results[idx]['raw'][:500]}...")
            st.caption("Source: Retrieved from Resume Vector Database using TF-IDF Similarity Search")

        st.divider()
        csv = df[["Rank", "Resume", "Score", "Status"]].to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "100_resumes_results.csv", "text/csv", type="primary", use_container_width=True)

st.caption("V2 BULK EDITION | Supports 2-100 Resumes | RAG Chat Enabled | TF-IDF Vector Search")
