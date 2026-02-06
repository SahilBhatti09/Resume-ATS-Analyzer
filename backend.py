"""
Backend Module for Resume-Job Description Matcher

This module handles all the text processing and similarity calculations
for comparing resumes with job descriptions using TF-IDF and Word2Vec methods.
"""

import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import numpy as np

# Download NLTK resources (comment out after first run)
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

# ------------------------
# Load resources
# ------------------------
# Load English stopwords (common words like 'the', 'is', 'at' that we ignore)
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer to convert words to their base form (e.g., 'running' -> 'run')
lemmatizer = WordNetLemmatizer()

# Load pre-trained Word2Vec model (Google News 300-dimensional vectors)
# This model understands word meanings and relationships
w2v_model = KeyedVectors.load("word2vec_google_news_300.kv")

# ------------------------
# PDF Extraction
# ------------------------
def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_file: Uploaded PDF file object from Streamlit
        
    Returns:
        str: Extracted text from all pages combined
    """
    text = ""
    # Open the PDF and loop through all pages
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

# ------------------------
# Preprocessing
# ------------------------
def clean_text(text):
    """
    Clean and preprocess text for analysis.
    
    Steps:
    1. Convert to lowercase
    2. Remove non-alphabetic characters
    3. Split into words (tokens)
    4. Remove stopwords (common words like 'the', 'is')
    5. Lemmatize (convert words to base form)
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        list: List of cleaned tokens
    """
    # Convert all text to lowercase
    text = text.lower()
    
    # Remove numbers, punctuation, special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Split text into individual words
    tokens = text.split()
    
    # Remove stopwords and lemmatize each word
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return tokens

# ------------------------
# TF-IDF Similarity + Top Keywords
# ------------------------
def tfidf_similarity_and_keywords(text1, text2, top_n=5):
    """
    Calculate exact word match similarity using TF-IDF and extract top keywords.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) measures how important
    a word is in a document by checking:
    - How often it appears in the document (TF)
    - How rare it is across all documents (IDF)
    
    Args:
        text1 (str): Resume text
        text2 (str): Job description text
        top_n (int): Number of top keywords to return (default: 5)
        
    Returns:
        tuple: (similarity_score, list_of_top_keywords)
            - similarity_score: Float between 0 and 1 (higher is better)
            - list_of_top_keywords: Top N keywords from resume
    """
    # Clean both texts and convert to document format
    documents = [" ".join(clean_text(text1)), " ".join(clean_text(text2))]
    
    # Create TF-IDF vectors for both documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity between resume and job description
    # Score ranges from 0 (no match) to 1 (perfect match)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Extract top keywords from resume based on TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    resume_vector = tfidf_matrix[0].toarray()[0]
    
    # Get indices of top N keywords (sorted by importance)
    indices = resume_vector.argsort()[::-1][:top_n]
    top_keywords = [feature_names[i] for i in indices]
    
    return score, top_keywords

# ------------------------
# Word2Vec Similarity
# ------------------------
def avg_word2vec(tokens):
    """
    Convert a list of words into a single average vector.
    
    Word2Vec represents each word as a 300-dimensional vector.
    Words with similar meanings have similar vectors.
    We average all word vectors to get a single vector for the entire text.
    
    Args:
        tokens (list): List of cleaned words
        
    Returns:
        numpy.array: 300-dimensional average vector
    """
    # Get vectors for words that exist in the Word2Vec model
    vectors = [w2v_model[w] for w in tokens if w in w2v_model]
    
    if vectors:
        # Calculate average of all word vectors
        return np.mean(vectors, axis=0)
    else:
        # Return zero vector if no words found in model
        return np.zeros(300)

def word2vec_similarity(text1, text2):
    """
    Calculate semantic similarity using Word2Vec embeddings.
    
    This method understands word meanings and relationships.
    For example, it knows that "Python programming" and "Python development"
    are related even though they use different words.
    
    Args:
        text1 (str): Resume text
        text2 (str): Job description text
        
    Returns:
        float: Similarity score between 0 and 1 (higher is better)
    """
    # Convert both texts to average Word2Vec vectors
    vec1 = avg_word2vec(clean_text(text1))
    vec2 = avg_word2vec(clean_text(text2))
    
    # Calculate cosine similarity between the two vectors
    # Score ranges from 0 (no similarity) to 1 (identical meaning)
    score = cosine_similarity([vec1], [vec2])[0][0]
    
    return score

# ------------------------
# Threshold Badge
# ------------------------
def get_match_badge(score):
    """
    Convert numerical score to human-readable status badge.
    
    Args:
        score (float): Similarity score between 0 and 1
        
    Returns:
        str: Status label based on score threshold
    """
    if score >= 0.75:
        return "Highly Suitable"
    elif score >= 0.5:
        return "Moderate"
    else:
        return "Low Match"
