import pdfplumber
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import ngrams
import re

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_terms(text, n=2):
    """Extract unigrams and bigrams from text for synonym mapping."""
    words = re.findall(r'\b\w+\b', text.lower())
    unigrams = words
    bigrams = [' '.join(gram) for gram in ngrams(words, 2)]
    return unigrams + bigrams

def create_dynamic_synonym_map(cv_text, jd_text, similarity_threshold=0.8):
    """Create a dynamic synonym map using SBERT to find similar terms."""
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    synonym_map = {}
    
    # Extract terms from CV and JD
    cv_terms = extract_terms(cv_text)
    jd_terms = extract_terms(jd_text)
    
    # Compute embeddings for terms
    cv_term_embeddings = model.encode(cv_terms, convert_to_tensor=True)
    jd_term_embeddings = model.encode(jd_terms, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(cv_term_embeddings, jd_term_embeddings)
    
    # Create synonym map for terms with high similarity
    for i, cv_term in enumerate(cv_terms):
        for j, jd_term in enumerate(jd_terms):
            if similarities[i][j] > similarity_threshold:
                synonym_map[cv_term] = jd_term  # Map CV term to JD term
                break  # Take the first high-similarity match to avoid redundancy
    
    return synonym_map

def preprocess_text(text, synonym_map):
    """Preprocess text by mapping CV terms to JD terms using the synonym map."""
    for synonym, standard in synonym_map.items():
        text = text.lower().replace(synonym, standard)
    return text

def compute_sbert_score(cv_text, jd_text):
    """Compute SBERT similarity score between CV and Job Description."""
    model = SentenceTransformer('all-roberta-large-v1')
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    similarity_score = util.cos_sim(cv_embedding, jd_embedding)[0][0].item()
    return similarity_score

def compute_tfidf_score(cv_text, jd_text):
    """Compute TF-IDF cosine similarity score between CV and Job Description."""
    vectorizer = TfidfVectorizer(stop_words=['the', 'is', 'and'], max_features=5000)
    texts = [cv_text, jd_text]
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_score = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
    return similarity_score

def compute_combined_score(cv_path, jd_path, sbert_weight=0.6, tfidf_weight=0.4, perfection_factor=1.8):
    """Compute combined SBERT and TF-IDF similarity score with grace score adjustment."""
    cv_text = extract_text_from_pdf(cv_path)
    jd_text = extract_text_from_pdf(jd_path)
    if not cv_text or not jd_text:
        print("Text extraction failed for one or both files.")
        return 0.0
    
    # Create dynamic synonym map
    synonym_map = create_dynamic_synonym_map(cv_text, jd_text)
    
    # Preprocess texts with synonym map
    cv_text = preprocess_text(cv_text, synonym_map)
    jd_text = preprocess_text(jd_text, synonym_map)
    
    # Compute individual scores
    sbert_score = compute_sbert_score(cv_text, jd_text)
    tfidf_score = compute_tfidf_score(cv_text, jd_text)
    
    # Compute weighted combined score
    combined_score = (sbert_weight * sbert_score) + (tfidf_weight * tfidf_score)
    
    # Apply grace score with perfection factor
    grace_score = min(combined_score * perfection_factor, 1.0)  # Cap at 100%
    score_percentage = round(grace_score * 100, 2)
    
    return score_percentage

def main():
    # Example usage
    cv_path = r"D:\Projects\student_placement_portal\jd_cv_scorer\cv_test\venkateshthanneru.pdf"
    jd_path = r"D:\Projects\student_placement_portal\jd_cv_scorer\jd_test\FSD.pdf"
    
    score = compute_combined_score(cv_path, jd_path)
    print(f"Combined SBERT + TF-IDF Similarity Score with Grace Adjustment: {score}%")

if __name__ == "__main__":
    main()
