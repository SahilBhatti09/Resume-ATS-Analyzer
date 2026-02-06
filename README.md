# 💼 Resume–Job Description Matcher

A simple web application that helps you compare your resume with a job description to see how well they match. Get instant feedback on your resume's alignment with the job requirements.

## 🌟 Features

- **Dual Input Options**: Upload PDF files or paste text directly
- **Two Scoring Methods**:
  - **Exact Word Match (TF-IDF)**: Checks how many exact keywords from the job description appear in your resume
  - **Skill Match (Word2Vec)**: Understands if your skills are related to the job, even if you use different words
- **Easy-to-Understand Results**: Get plain English explanations of your scores
- **Top Keywords**: See the most important matching keywords in your resume
- **Visual Comparison**: Bar chart showing both match scores side-by-side
- **Actionable Tips**: Get suggestions on how to improve your resume

## 🎯 How It Works

The app uses two different methods to analyze your resume:

### 1. Exact Word Match (TF-IDF)
- Looks for **exact matching words** between your resume and the job description
- **Example**: If the job asks for "Python" and you wrote "Python", that's a match
- **Low score?** You might need to use more keywords from the job description

### 2. Skill Match (Word2Vec)
- Understands **similar meanings** even with different words
- **Example**: If the job asks for "Python programming" and you wrote "Python development", it knows they're related
- **High score?** You have the right skills, even if worded differently

### Why Two Scores?

Both scores tell you different things:
- **High Exact Match + High Skill Match** = Perfect! Your resume is well-aligned
- **Low Exact Match + High Skill Match** = You have the right skills, but use more job description keywords
- **High Exact Match + Low Skill Match** = You're using the right words, but highlight more relevant experience
- **Low Both** = Consider tailoring your resume more closely to the job

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Word2Vec model file (`word2vec_google_news_300.kv`)

### Steps

1. **Clone or download this repository**
```bash
git clone https://github.com/SahilBhatti09/Resume-ATS-Analyzer.git
cd Resume-ATS-Analyzer
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (first time only)
   - Uncomment lines 17-19 in `backend.py` and run the app once
   - After the first run, comment those lines back to avoid re-downloading

4. **Get the Word2Vec model** (Required - Not included in repo due to size)
   - Download the pre-trained Google News Word2Vec model
   - You can download it from: [Google News Word2Vec](https://github.com/eyaler/word2vec-slim)
   - Or use this command to download and convert:
   ```bash
   # Install gensim if not already installed
   pip install gensim
   
   # Download and convert the model (this will take time - ~3GB)
   python -c "import gensim.downloader as api; model = api.load('word2vec-google-news-300'); model.save('word2vec_google_news_300.kv')"
   ```
   - Place the `word2vec_google_news_300.kv` and `word2vec_google_news_300.kv.vectors.npy` files in the project folder

## 📖 Usage

### Running the Application

1. **Start the app**
```bash
streamlit run frontend.py
```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

3. **Choose your input method** for Resume:
   - Option 1: Upload PDF file
   - Option 2: Paste text directly

4. **Choose your input method** for Job Description:
   - Option 1: Upload PDF file
   - Option 2: Paste text directly

5. **Click "Check Match"** to see your results

### Understanding Your Results

The app will show you:

- **📊 Match Scores**: Two percentage scores showing your match quality
- **📝 What This Means**: Plain English explanation of your scores
- **💡 Tips**: Actionable suggestions to improve your resume
- **🌟 Top Keywords**: The most important matching keywords found
- **📈 Visual Comparison**: Bar chart comparing both scores

## 📁 Project Structure

```
resume-match/
│
├── frontend.py              # Streamlit web interface
├── backend.py               # Processing logic and NLP models
├── requirements.txt         # Python dependencies
├── word2vec_google_news_300.kv  # Pre-trained Word2Vec model
├── README.md               # This file
└── sample outputs          # use case testing screenshots
```

## 🛠️ Technologies Used

- **Streamlit**: Web interface
- **NLTK**: Text preprocessing (stopwords, lemmatization)
- **scikit-learn**: TF-IDF vectorization and similarity calculation
- **Gensim**: Word2Vec model loading and similarity
- **pdfplumber**: PDF text extraction
- **matplotlib**: Data visualization
- **NumPy**: Numerical operations

## 📊 Score Interpretation

| Score Range | Status | Meaning |
|-------------|--------|---------|
| 75% - 100% | Highly Suitable | Strong match with the job requirements |
| 50% - 74% | Moderate | Decent match, room for improvement |
| 0% - 49% | Low Match | Consider tailoring your resume more |

## 💡 Tips for Best Results

1. **Use clear formatting**: Make sure your PDF or text is readable
2. **Include relevant keywords**: Use terms from the job description naturally
3. **Highlight skills**: Clearly list your technical and soft skills
4. **Be specific**: Use concrete examples of your experience
5. **Match the language**: If the job uses "Python", use "Python" (not just "programming")

## ⚠️ Notes

- The app works best with English text
- Very short resumes or job descriptions may give less accurate results
- **The Word2Vec model files (~3GB) are NOT included in this repository** due to GitHub file size limits
  - You must download them separately (see Installation step 4)
  - Both `.kv` and `.vectors.npy` files are required
- After first run, comment out the NLTK download lines in `backend.py` (lines 17-19)
- The `.gitignore` file excludes the model files to keep the repository lightweight

## 🤝 Contributing

Feel free to fork this project and make improvements. Some ideas:
- Add support for more file formats (Word, txt)
- Add more visualization options
- Include industry-specific keyword suggestions
- Add resume improvement suggestions

## 📝 License

This project is open source and available for educational purposes.

## Author

Sahil Bhatti

---

**Made with ❤️ using Python and Streamlit**
