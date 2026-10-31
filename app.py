import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Screener Pro V2 RAG", layout="wide", page_icon="🏆")

st.title("🏆 AI Resume Screener Pro V2 - 100 Resumes + RAG!")
st.markdown("### Upload 2 to 100 Resumes - AI ranks + Chat with Resumes!")
st.divider()

with st.sidebar:
    st.header("⚙️ Bulk Settings")
    threshold = st.slider("Selection Threshold %", 0, 100, 15)
    st.success("✅ Supports up to 100 resumes!")
    st.info("NEW: RAG Chat Added!")

st.subheader("📋 Job Description")
jd = st.text_area("Paste JD", height=180)

st.subheader("📄 Upload Resumes (2 to 100)")
uploaded = st.file_uploader(f"Select up to 100 PDFs", accept_multiple_files=True, type=['pdf'])

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

if st.button("🔍 SCREEN 100 RESUMES + RAG NOW", type="primary", use_container_width=True):
    if not jd:
        st.error("JD paste pannu da!")
    elif not uploaded:
        st.error("Resume upload pannu da!")
    elif len(uploaded) > 100:
        st.error("⚠️ Max 100 resumes only da!")
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

        progress.progress(0.9, text="AI Ranking all resumes...")
        docs = [jd_clean] + [r["clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform(docs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = []
        for i, r in enumerate(resumes_data):
            score = round(float(scores[i]) * 100, 2)
            status = "✅ SELECTED" if score >= threshold else "❌ REJECTED"
            results.append({"Resume": r["name"], "Score": score, "Status": status, "raw": r["raw"], "clean": r["clean"]})

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df)+1))
        progress.empty()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df))
        c2.metric("Selected", len(df[df['Score']>=threshold]))
        c3.metric("Rejected", len(df[df['Score']<threshold]))
        c4.metric("Top Score", f"{df['Score'].max()}%")

        st.divider()
        st.subheader(f"🏆 Leaderboard - Top {len(df)} Resumes Ranked")
        st.dataframe(df[["Rank", "Resume", "Score", "Status"]], use_container_width=True, hide_index=True, height=600)

        # ===== RAG CHAT WITHOUT LANGCHAIN =====
        st.divider()
        st.subheader("💬 RAG Chat - Chat with 100 Resumes!")
        st.success("RAG Ready! Kelvi kelu da!")
        
        query = st.text_input("Ex: Why Rank 1 is best? Who has Python? Compare top 2")
        if query:
            q_clean = clean_text(query)
            rag_docs = [r["clean"] for r in results] + [q_clean]
            rag_vectorizer = TfidfVectorizer(stop_words='english')
            rag_tfidf = rag_vectorizer.fit_transform(rag_docs)
            rag_scores = cosine_similarity(rag_tfidf[-1:], rag_tfidf[:-1]).flatten()
            
            # Top 3 relevant
            top_idx = rag_scores.argsort()[::-1][:3]
            st.write(f"**RAG Answer for:** {query}")
            for idx in top_idx:
                if rag_scores[idx] > 0.05:
                    st.markdown(f"- **{results[idx]['Resume']} ({results[idx]['Score']}%) - Relevance {round(rag_scores[idx]*100,1)}%**: {results[idx]['raw'][:350]}...")
            st.caption("Source: TF-IDF Vector DB | Proof retrieved from resume text - No Langchain needed!")

        csv = df[["Rank", "Resume", "Score", "Status"]].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full 100 Results CSV", csv, "100_resumes_rag_leaderboard.csv", "text/csv", type="primary")

st.caption("V2 BULK EDITION | 2-100 resumes | RAG Chat | Error-Free | No Langchain")
