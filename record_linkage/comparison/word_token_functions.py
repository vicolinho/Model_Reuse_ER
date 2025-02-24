import numpy as np


idf_map = None

def compute_tf_idf(term_freq):
    """Calculate the TF-IDF vector for a given term frequency dictionary and IDF map."""
    tf_idf_vector = {}
    for term, tf in term_freq.items():
        if term in idf_map:
            tf_idf_vector[term] = tf * idf_map[term]
    return tf_idf_vector

def cosine_tfidf_similarity(word_count1, word_count2):
    """Calculate cosine similarity between two TF-IDF vectors."""
    # Create a set of all terms in both vectors
    vec1 = compute_tf_idf(word_count1)
    vec2 = compute_tf_idf(word_count2)
    all_terms = set(vec1.keys()).union(set(vec2.keys()))

    # Create the corresponding vector representations
    vector1 = np.array([vec1.get(term, 0) for term in all_terms])
    vector2 = np.array([vec2.get(term, 0) for term in all_terms])

    # Calculate cosine similarity
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # Avoid division by zero

    return dot_product / (norm1 * norm2)