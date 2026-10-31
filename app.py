import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Screener Pro", layout="wide", page_icon="🏆")

st.title("🏆 AI Resume Screener Pro - BULK 50 Supported!")
st.markdown("### Upload 2 to 50 Resumes at once - AI ranks all!")
st.divider()

with st.sidebar:
    st.header("⚙️ Bulk Settings")
    threshold = st.slider("Selection Threshold %", 0, 100, 60)
    st.success("✅ Supports up to 50 resumes at once!")
    st.info("Tip: 50 resumes ~ 30 sec processing")

st.subheader("📋 Job Description")
jd = st.text_area("Paste JD", height=180)

st.subheader("📄 Upload Resumes (2 to 50)")
uploaded = st.file_uploader(f"Select up to 50 PDFs", accept_multiple_files=True, type=['pdf'])

def extract_text(pdf):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf)
        for page in reader.pages[:5]:  # First 5 pages only for speed - 50 resume fast
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

if st.button("🔍 SCREEN 50 RESUMES NOW", type="primary", use_container_width=True):
    if not jd:
        st.error("JD paste pannu da!")
    elif not uploaded:
        st.error("Resume upload pannu da!")
    elif len(uploaded) > 50:
        st.error("⚠️ Max 50 resumes only da! 50-ku mela upload panna koodathu")
    else:
        progress = st.progress(0, text="Reading resumes...")
        jd_clean = clean_text(jd)
        resumes_data = []
        
        for idx, file in enumerate(uploaded):
            progress.progress((idx+1)/len(uploaded), text=f"Reading {idx+1}/{len(uploaded)}: {file.name}")
            raw = extract_text(file)
            if not raw:
                raw = file.name
            resumes_data.append({"name": file.name, "clean": clean_text(raw)})

        progress.progress(0.9, text="AI Ranking all resumes...")
        docs = [jd_clean] + [r["clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) # Fast for 50
        tfidf = vectorizer.fit_transform(docs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = []
        for i, r in enumerate(resumes_data):
            score = round(float(scores[i]) * 100, 2)
            status = "✅ SELECTED" if score >= threshold else "❌ REJECTED"
            results.append({"Resume": r["name"], "Score": score, "Status": status})

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
        st.dataframe(df, use_container_width=True, hide_index=True, height=600)

        st.divider()
        st.subheader("🎖️ Top 3 Winners")
        for idx, row in df.head(10).iterrows():
            if idx == 0:
                st.success(f"🥇 RANK 1 - {row['Resume']} - {row['Score']}%")
            elif idx == 1:
                st.warning(f"🥈 RANK 2 - {row['Resume']} - {row['Score']}%")
            elif idx == 2:
                st.info(f"🥉 RANK 3 - {row['Resume']} - {row['Score']}%")
            else:
                st.write(f"**{row['Rank']}.** {row['Resume']} - {row['Score']}% {row['Status']}")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full 50 Results CSV", csv, "50_resumes_leaderboard.csv", "text/csv", type="primary")

st.caption("BULK EDITION | 2-50 resumes | Progress bar | Fast processing")
