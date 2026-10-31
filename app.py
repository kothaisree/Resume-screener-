import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
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
        raw_texts_for_rag = []
        
        for idx, file in enumerate(uploaded):
            progress.progress((idx+1)/len(uploaded), text=f"Reading {idx+1}/{len(uploaded)}: {file.name}")
            raw = extract_text(file)
            if not raw:
                raw = file.name
            resumes_data.append({"name": file.name, "clean": clean_text(raw), "raw": raw})
            raw_texts_for_rag.append(raw)

        progress.progress(0.9, text="AI Ranking all resumes...")
        docs = [jd_clean] + [r["clean"] for r in resumes_data]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform(docs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = []
        for i, r in enumerate(resumes_data):
            score = round(float(scores[i]) * 100, 2)
            status = "✅ SELECTED" if score >= threshold else "❌ REJECTED"
            results.append({"Resume": r["name"], "Score": score, "Status": status, "raw": r["raw"]})

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

        # RAG SECTION
        st.divider()
        st.subheader("💬 RAG Chat - Chat with 100 Resumes!")
        
        @st.cache_resource
        def build_rag(_texts, _names):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            documents = [Document(page_content=t, metadata={"name": n}) for t, n in zip(_texts, _names)]
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            return FAISS.from_documents(chunks, embeddings)

        with st.spinner("Building RAG Vector DB..."):
            names = [r["Resume"] for r in results]
            vectorstore = build_rag(raw_texts_for_rag, names)
        
        st.success("RAG Ready! Kelvi kelu da!")
        query = st.text_input("Ex: Why Rank 1 is best? Who has Python?")
        if query:
            retrieved = vectorstore.similarity_search(query, k=5)
            st.write(f"**Answer for:** {query}")
            for doc in retrieved:
                st.markdown(f"- **{doc.metadata['name']}**: {doc.page_content[:350]}...")
            st.caption("Source: FAISS + HuggingFace Embeddings | Proof from resume")

        csv = df[["Rank", "Resume", "Score", "Status"]].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full 100 Results CSV", csv, "100_resumes_rag_leaderboard.csv", "text/csv", type="primary")

st.caption("V2 BULK EDITION | 2-100 resumes | RAG Chat | Company Ready")
