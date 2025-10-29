import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine
from urllib.parse import quote

# ----------------------------
# Configuration
# ----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = r"D:\\360DigitMG\\Project - 2\\ChatBot Code\\vector_database.index"
DOCUMENTS_PATH = r"D:\\360DigitMG\\Project - 2\\ChatBot Code\\metadata.json"
GEMINI_API_KEY = "AIzaSyDaoMJXZsSmHzgmXAfTctQU5u_UzmABG3g"
LOGO_PATH = r"D:\\360DigitMG\\Project - 2\\ChatBot Code\\AiSPRY logo.jpg"
DB_USER = "root"
DB_PASSWORD = "root"
DB_NAME = "ai_chatbot"


# ----------------------------
# Initialize Database Engine
# ----------------------------
def get_db_engine():
    return create_engine(f"mysql+pymysql://{DB_USER}:%s@localhost/{DB_NAME}" % quote(f'{DB_PASSWORD}'))


# ----------------------------
# Load Embedding Model, FAISS Index & Documents
# ----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    return {"index": index}

@st.cache_resource
def load_documents():
    with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# ----------------------------
# Initialize Gemini Model
# ----------------------------
def initialize_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("models/gemini-1.5-flash")

# ----------------------------
# Save query to database using pandas
# ----------------------------
def save_query_to_db(query, engine):
    # Create a DataFrame with query and timestamp
    query_data = pd.DataFrame({
        'query': [query],
        'timestamp': [datetime.utcnow()]
    })
    
    # Insert the data into the 'user_queries' table
    query_data.to_sql('user_queries', con=engine, if_exists='append', index=False)

# ----------------------------
# Search Function
# ----------------------------
def search_faiss(query, model, faiss_data, documents, top_k=5):
    query_embedding = model.encode([query])
    index = faiss_data['index']
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    retrieved_texts = [documents[i]['text'] for i in indices[0] if i < len(documents) and 'text' in documents[i]]
    return retrieved_texts

# ----------------------------
# Streamlit UI Starts
# ----------------------------
def main():
    st.set_page_config(page_title="Class X EduBot", layout="wide")

    # Sidebar
    with st.sidebar:
        st.image(LOGO_PATH, use_container_width=True)
        st.markdown("""
            <div style='padding-top: 10px;'>
                <h3 style='color:#003366;'>EduBot - Class X Science</h3>
                <hr style='border: 1px solid #003366;'>
                <p style='font-size: 14px;'>Powered by AiSPRY</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color:#f0f8ff;padding:20px 30px;border-radius:12px;margin-bottom:25px">
            <h2 style="color:#003366;">📘 AI-powered Science Chatbot for Class X</h2>
            <p style="font-size:16px;">Ask your science questions below based on the Class X syllabus!</p>
        </div>
    """, unsafe_allow_html=True)

    # Load resources
    model = load_embedding_model()
    faiss_data = load_faiss_index()
    documents = load_documents()
    gemini = initialize_gemini()

    # Chat input UI
    user_query = st.text_input("🔍 Enter your science question:", placeholder="e.g., What is photosynthesis?")
    
    engine = get_db_engine()
    save_query_to_db(user_query, engine)

    if user_query:
        with st.spinner("🔎 Searching and generating answer..."):
            retrieved_context = search_faiss(user_query, model, faiss_data, documents, top_k=5)
            combined_context = "\n\n".join(retrieved_context)

            prompt = f"""
You are an educational assistant chatbot for Class X Science. Answer the following question based on the provided context. 
If possible, also describe what a helpful diagram or image would show for better understanding of the topic.
If the context is not sufficient, politely say that you don’t have enough information.

Context:
{combined_context}

Question: {user_query}

Answer:
"""
            response = gemini.generate_content(prompt)

        st.markdown("""
            <div style="background-color:#e0f7fa;padding:20px;border-radius:10px;margin-top:20px;">
                <h4 style="color:#00796B;">✅ Answer:</h4>
                <p style="font-size:16px;">{}</p>
            </div>
        """.format(response.text), unsafe_allow_html=True)

        with st.expander("🔍 Show retrieved context"):
            for idx, text in enumerate(retrieved_context):
                st.markdown(f"**Context {idx+1}:**\n{text}")

if __name__ == '__main__':
    main()


