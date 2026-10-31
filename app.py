import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Resume Screener Pro", page_icon="🚀", layout="wide")

# --- CUSTOM CSS FOR PRO LOOK ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .metric-card { background: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🚀 AI Resume Screener <span style='color:white;'>PRO</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>AI-Powered ATS | Rank 50 Resumes in 10 Seconds | Built with Python & ML</p>", unsafe_allow_html=True)
st.divider()

# --- INPUTS ---
col1, col2 = st.columns([1, 1.5])
with col1:
    st.subheader("📋 Job Description")
    jd = st.text_area("Paste JD here", height=300, placeholder="Ex: Looking for Python Developer with ML, SQL experience...")
    threshold = st.slider("Selection Threshold %", 0, 100, 60)

with col2:
    st.subheader("📄 Upload Resumes (PDF)")
    files = st.file_uploader("Drag & Drop", type="pdf", accept_multiple_files=True)

# --- LOGIC ---
def extract_text(pdf):
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

if st.button("🔍 SCREEN RESUMES NOW", use_container_width=True, type="primary"):
    if not jd or not files:
        st.warning("Please add JD and upload resumes!")
    else:
        with st.spinner("AI is screening..."):
            resumes_text = [extract_text(f) for f in files]
            names = [f.name for f in files]
            
            docs = [jd] + resumes_text
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(docs)
            
            jd_vec = vectors[0]
            res_vecs = vectors[1:]
            
            scores = cosine_similarity(jd_vec, res_vecs)[0] * 100
            
            data = []
            for i, score in enumerate(scores):
                status = "✅ SELECTED" if score >= threshold else "❌ REJECTED"
                badge = "🏆 TOP CANDIDATE" if score == max(scores) else ""
                data.append({"Rank": 0, "Resume": names[i], "Match %": round(score, 2), "Status": status, "Badge": badge})
            
            df = pd.DataFrame(data)
            df = df.sort_values(by="Match %", ascending=False)
            df['Rank'] = range(1, len(df)+1)
            
            # --- RESULTS ---
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Resumes", len(files))
            c2.metric("Selected", len(df[df['Match %']>=threshold]))
            c3.metric("Top Score", f"{max(scores):.2f}%")
            
            st.subheader("🏆 Leaderboard")
            
            # Color function
            def color_score(val):
                if val >= 75: color = "#22c55e"
                elif val >= 60: color = "#eab308"
                else: color = "#ef4444"
                return f'background-color: {color}; color: white; font-weight: bold;'
            
            styled_df = df.style.applymap(color_score, subset=['Match %'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Result as Excel/CSV", csv, "screening_result.csv", "text/csv", use_container_width=True)

st.markdown("<p style='text-align:center; color: #475569; margin-top: 50px;'>Built by Kothai | Python + TF-IDF + Cosine Similarity</p>", unsafe_allow_html=True)
