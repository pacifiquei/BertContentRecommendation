import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch

# Load the model from Hugging Face
@st.cache_resource
def load_model():
    return SentenceTransformer("pacifiquei/BertContentRecommendation")

# Load real content data with precomputed embeddings
@st.cache_data
def load_content_data():
    content_data = pd.read_pickle("content_data_embeddings.pkl")
    # Convert embeddings from list format to tensor for cosine similarity
    content_data['embedding'] = content_data['embedding'].apply(torch.tensor)
    return content_data

# Initialize model and data
model = load_model()
content_data = load_content_data()

# Streamlit UI
st.title("Product Recommendation System")
search_term = st.text_input("Enter the product name:")
top_k = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):
    if search_term:
        # Generate embedding for the search term
        search_embedding = model.encode([search_term])[0]

        # Calculate cosine similarity between search term and content data embeddings
        similarity_scores = cosine_similarity([search_embedding], list(content_data['embedding'].values))
        similarity_scores = similarity_scores.flatten()

        # Get top-k most similar content titles
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        recommended_titles = content_data.iloc[top_indices]

        # Display recommendations
        st.subheader("Recommendations:")
        for title in recommended_titles['title']:
            st.write(f"- {title}")
    else:
        st.warning("Please enter a product name.")
