import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="🏆 AI Resume Screener Pro", layout="wide", page_icon="🏆")

if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.df = None

# ===== PREMIUM CSS DESIGN =====
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.main-header h1 {
    font-size: 42px !important;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.main-header p {
    font-size: 18px;
    opacity: 0.9;
    margin-top: 8px;
}
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}
.stButton>button {
    border-radius: 25px !important;
    height: 50px !important;
    font-weight: bold !important;
    font-size: 18px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%) !important;
    border: none !important;
}
.rank-1 { background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%); padding: 15px; border-radius: 12px; color: #000; font-weight: bold; margin: 5px 0; }
.rank-2 { background: linear-gradient(90deg, #C0C0C0 0%, #A8A8A8 100%); padding: 15px; border-radius: 12px; color: #000; font-weight: bold; margin: 5px 0; }
.rank-3 { background: linear-gradient(90deg, #CD7F32 0%, #A0522D 100%); padding: 15px; border-radius: 12px; color: white; font-weight: bold; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# ===== HEADER WITH 🏆 =====
st.markdown("""
<div class="main-header">
    <h1>🏆 AI Resume Screener Pro V2</h1>
    <p>✨ Bulk Screening | 2 to 100 Resumes | RAG Chat | Vector Search | Premium Design</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("🎯 Selection Threshold %", 0, 100, 15)
    st.markdown("---")
    st.success("✅ Supports up to 100 resumes")
    st.info("🚀 Features:\n- 🏆 Premium UI\n- 📊 Bulk Ranking\n- 💬 RAG Chat\n- 🔍 Vector Search")
    if st.button("🗑️ Clear Results"):
        st.session_state.results = None
        st.session_state.df = None
        st.rerun()

st.markdown("### 📝 Paste Job Description Here")
jd = st.text_area("", height=180, placeholder="Ex: Python Developer with Flask, MySQL, JavaScript, 2+ years, Chennai...", label_visibility="collapsed")

st.markdown("### 📤 Upload Resumes (2 to 100 PDFs)")
uploaded = st.file_uploader("Select PDFs", accept_multiple_files=True, type=['pdf'], label_visibility="collapsed")

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

def extract_name(raw_text):
    lines = raw_text.strip().split('\n')
    if lines and len(lines[0]) < 50 and '@' not in lines[0] and len(lines[0].split()) <= 4:
        return lines[0].strip()
    return "Unknown Candidate"

if st.button("🚀 SCREEN 100 RESUMES NOW", type="primary", use_container_width=True):
    if not jd:
        st.error("❌ Please paste Job Description")
    elif not uploaded:
        st.error("❌ Please upload resumes")
    elif len(uploaded) > 100:
        st.error("❌ Maximum 100 resumes allowed")
    else:
        progress = st.progress(0, text="📖 Reading resumes...")
        jd_clean = clean_text(jd)
        resumes_data = []
        
        for idx, file in enumerate(uploaded):
            progress.progress((idx+1)/len(uploaded), text=f"📄 Reading {idx+1}/{len(uploaded)}: {file.name}")
            raw = extract_text(file)
            if not raw:
                raw = file.name
            resumes_data.append({"name": file.name, "clean": clean_text(raw), "raw": raw})

        docs = [jd_clean] + [r["clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform(docs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = []
        for i, r in enumerate(resumes_data):
            score = round(float(scores[i]) * 100, 2)
            status = "SELECTED" if score >= threshold else "REJECTED"
            results.append({"Candidate Name": extract_name(r["raw"]), "Resume": r["name"], "Score": score, "Status": status, "raw": r["raw"], "clean": r["clean"]})

        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df)+1))
        
        st.session_state.results = results
        st.session_state.df = df
        progress.empty()
        st.balloons()

if st.session_state.results is not None:
    df = st.session_state.df
    results = st.session_state.results
    
    st.markdown("---")
    st.markdown("### 📊 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Total Resumes", len(df))
    c2.metric("✅ Selected", len(df[df['Score']>=threshold]))
    c3.metric("❌ Rejected", len(df[df['Score']<threshold]))
    c4.metric("🏆 Top Score", f"{df['Score'].max()}%")

    st.markdown("---")
    st.markdown(f"### 🏆 Leaderboard - Top {len(df)} Resumes Ranked")
    st.dataframe(df[["Rank", "Candidate Name", "Resume", "Score", "Status"]], use_container_width=True, hide_index=True, height=450)

    st.markdown("### 🎖️ Top 3 Winners")
    for idx, row in df.head(3).iterrows():
        if idx == 0:
            st.markdown(f'<div class="rank-1">🥇 RANK 1 - {row["Candidate Name"]} | {row["Resume"]} - {row["Score"]}% 🏆</div>', unsafe_allow_html=True)
        elif idx == 1:
            st.markdown(f'<div class="rank-2">🥈 RANK 2 - {row["Candidate Name"]} | {row["Resume"]} - {row["Score"]}%</div>', unsafe_allow_html=True)
        elif idx == 2:
            st.markdown(f'<div class="rank-3">🥉 RANK 3 - {row["Candidate Name"]} | {row["Resume"]} - {row["Score"]}%</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 RAG Chat - Chat with 100 Resumes")
    st.success("✅ 🏆 RAG System Ready - Vector DB Built Successfully | Ask Anything!")

    query = st.text_input("🔍 Ask a question", placeholder="Ex: Why is Rank 1 best? Who has Python? Compare top 2")
    
    if query:
        q_clean = clean_text(query)
        rag_docs = [r["clean"] for r in results] + [q_clean]
        rag_vectorizer = TfidfVectorizer(stop_words='english')
        rag_tfidf = rag_vectorizer.fit_transform(rag_docs)
        rag_scores = cosine_similarity(rag_tfidf[-1:], rag_tfidf[:-1]).flatten()
        
        top_idx = rag_scores.argsort()[::-1][:3]
        st.markdown(f"#### 🔍 Answer for: **{query}**")
        for idx in top_idx:
            if rag_scores[idx] > 0.03:
                st.markdown(f"**🏅 {results[idx]['Candidate Name']} | {results[idx]['Resume']} ({results[idx]['Score']}%) - Relevance {round(rag_scores[idx]*100,1)}%**")
                with st.expander(f"📄 View {results[idx]['Candidate Name']} details", expanded=(idx==top_idx[0])):
                    st.info(f"{results[idx]['raw'][:800]}...")

    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV (with Names)", csv, "🏆_results.csv", "text/csv", type="primary", use_container_width=True)

st.markdown("<center><small>🏆 V2 BULK EDITION | Premium Design | Supports 2-100 Resumes | RAG Chat Enabled | Made with ❤️</small></center>", unsafe_allow_html=True)
