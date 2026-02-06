"""
Frontend Module for Resume-Job Description Matcher

This is the Streamlit web interface that allows users to upload resumes
and job descriptions (as PDF or text) and see match analysis results.
"""

import streamlit as st
from backend import extract_text_from_pdf, tfidf_similarity_and_keywords, word2vec_similarity, get_match_badge
import matplotlib.pyplot as plt

def get_remarks(tfidf_score, w2v_score):
    """
    Generate easy-to-understand remarks and tips based on match scores.
    
    Args:
        tfidf_score (float): Exact word match score (0-1)
        w2v_score (float): Skill match score (0-1)
        
    Returns:
        list: List of remark strings to display to the user
    """
    # Convert scores to percentages for easier comparison
    tfidf_pct = tfidf_score * 100
    w2v_pct = w2v_score * 100
    
    remarks = []
    
    # Explain TF-IDF (Exact Word Match) score
    if tfidf_pct >= 75:
        remarks.append("✅ Your resume has many exact words from the job description.")
    elif tfidf_pct >= 50:
        remarks.append("⚠️ Your resume has some matching words with the job description.")
    else:
        remarks.append("❌ Your resume has few exact words from the job description.")
    
    # Explain Word2Vec (Skill Match) score
    if w2v_pct >= 75:
        remarks.append("✅ Your skills and experience are strongly related to the job requirements.")
    elif w2v_pct >= 50:
        remarks.append("⚠️ Your skills and experience are somewhat related to the job requirements.")
    else:
        remarks.append("❌ Your skills and experience are weakly related to the job requirements.")
    
    # Provide actionable recommendations based on score combinations
    if tfidf_pct < 50 and w2v_pct >= 75:
        remarks.append("💡 Tip: Try adding more keywords from the job description to your resume.")
    elif tfidf_pct >= 50 and w2v_pct < 50:
        remarks.append("💡 Tip: Consider highlighting more relevant skills and experiences.")
    elif tfidf_pct >= 75 and w2v_pct >= 75:
        remarks.append("🎉 Great match! Your resume aligns well with the job description.")
    elif tfidf_pct < 50 and w2v_pct < 50:
        remarks.append("⚠️ Consider tailoring your resume more closely to this job description.")
    
    return remarks

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="Resume–Job Description Matcher", page_icon="💼")

st.title("💼 Resume–Job Description Matcher")
st.write("Upload PDFs or paste text directly to see match scores, top keywords, and suitability badge.")

# ------------------------
# Resume Input Section
# ------------------------
st.subheader("Resume")

# Let user choose between uploading PDF or pasting text
resume_input_type = st.radio("Choose input method for Resume:", ["Upload PDF", "Paste Text"], key="resume")
resume_file = None
resume_text_input = None

# Show appropriate input widget based on user's choice
if resume_input_type == "Upload PDF":
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_file")
else:
    resume_text_input = st.text_area("Paste Resume Text:", height=200, key="resume_text")

# ------------------------
# Job Description Input Section
# ------------------------
st.subheader("Job Description")

# Let user choose between uploading PDF or pasting text
jd_input_type = st.radio("Choose input method for Job Description:", ["Upload PDF", "Paste Text"], key="jd")
jd_file = None
jd_text_input = None

# Show appropriate input widget based on user's choice
if jd_input_type == "Upload PDF":
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd_file")
else:
    jd_text_input = st.text_area("Paste Job Description Text:", height=200, key="jd_text")

# ------------------------
# Main Processing Logic
# ------------------------
if st.button("Check Match"):
    # Get resume text (either from PDF or pasted text)
    if resume_input_type == "Upload PDF":
        if resume_file is None:
            st.error("Please upload resume PDF.")
            st.stop()
        resume_text = extract_text_from_pdf(resume_file)
    else:
        if not resume_text_input or resume_text_input.strip() == "":
            st.error("Please paste resume text.")
            st.stop()
        resume_text = resume_text_input
    
    # Get job description text (either from PDF or pasted text)
    if jd_input_type == "Upload PDF":
        if jd_file is None:
            st.error("Please upload job description PDF.")
            st.stop()
        jd_text = extract_text_from_pdf(jd_file)
    else:
        if not jd_text_input or jd_text_input.strip() == "":
            st.error("Please paste job description text.")
            st.stop()
        jd_text = jd_text_input

    # Show loading message (model is already pre-loaded)
    with st.spinner("Loading Word2Vec model..."):
        pass  # Model is already loaded in backend.py

    # Calculate TF-IDF similarity and extract top keywords
    tfidf_score, top_keywords = tfidf_similarity_and_keywords(resume_text, jd_text)
    
    # Calculate Word2Vec semantic similarity
    w2v_score = word2vec_similarity(resume_text, jd_text)
    
    # Convert scores to status badges (Highly Suitable, Moderate, Low Match)
    tfidf_badge = get_match_badge(tfidf_score)
    w2v_badge = get_match_badge(w2v_score)
    
    # ------------------------
    # Display Results Section
    # ------------------------
    st.subheader("📊 Match Scores")
    
    # Create two columns for side-by-side score display
    col1, col2 = st.columns(2)
    
    # Display Exact Word Match score (TF-IDF)
    with col1:
        st.metric(
            label="Exact Word Match",
            value=f"{tfidf_score*100:.1f}%",
            help="How many exact words from job description appear in your resume"
        )
        st.caption(f"Status: **{tfidf_badge}**")
    
    # Display Skill Match score (Word2Vec)
    with col2:
        st.metric(
            label="Skill Match",
            value=f"{w2v_score*100:.1f}%",
            help="How well your skills match the job, even if you use different words"
        )
        st.caption(f"Status: **{w2v_badge}**")
    
    # ------------------------
    # Remarks and Explanations
    # ------------------------
    st.subheader("📝 What This Means")
    remarks = get_remarks(tfidf_score, w2v_score)
    for remark in remarks:
        st.write(remark)
    
    # ------------------------
    # Top Keywords Section
    # ------------------------
    st.subheader("🌟 Top Matching Keywords in Resume")
    st.write(", ".join(top_keywords))
    
    # ------------------------
    # Visual Comparison Chart
    # ------------------------
    st.subheader("📈 Visual Comparison")
    
    # Create bar chart with custom size
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(["Exact Word Match", "Skill Match"], [tfidf_score*100, w2v_score*100], color=['skyblue','orange'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Match Score (%)")
    ax.set_title("Your Resume Match Analysis")
    
    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Display the chart
    st.pyplot(fig)
