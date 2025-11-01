import streamlit as st
from src.rag_pipeline import retrieve_and_answer
import time

# Page Configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit's deploy menu */
    .stDeployButton {
        display: none;
    }
    
    /* Hide the hamburger menu */
    .stApp > header {
        visibility: hidden;
    }
    
    /* Hide the main menu */
    .stApp > header > div {
        visibility: hidden;
    }
    
    /* Hide the deploy button specifically */
    [data-testid="stHeader"] {
        display: none;
    }
    
    /* Hide the menu button */
    [data-testid="stToolbar"] {
        display: none;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 30px;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .feature-icon {
        font-size: 2rem;
        color: white;
        margin-bottom: 10px;
    }
    
    .feature-text {
        color: white;
        font-size: 1rem;
    }
    
    .chat-container {
        background: rgba(255,255,255,0.95);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    
    .answer-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-title">
    🏥 Medical AI Assistant
</div>
<div class="subtitle">
    Your intelligent medical question-answering assistant powered by RAG technology
</div>
""", unsafe_allow_html=True)

# Features using Streamlit columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-text">AI-Powered Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <div class="feature-text">Medical Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-text">Instant Responses</div>
    </div>
    """, unsafe_allow_html=True)

# Chat Interface
st.markdown("""
<div class="chat-container">
    <h3 style="color: #2c3e50; margin-bottom: 20px;">
        💭 Ask your medical question:
    </h3>
</div>
""", unsafe_allow_html=True)

# Input
query = st.text_input(
    "Medical Question",
    placeholder="Enter your medical question here...",
    label_visibility="collapsed"
)

# Button
if st.button("🔍 Get Medical Answer", type="primary"):
    if query.strip():
        with st.spinner("Processing your question..."):
            time.sleep(0.5)
            result = retrieve_and_answer(query, top_k=2)
            
            st.markdown(f"""
            <div class="answer-box">
                <h4 style="color: #2c3e50; margin-bottom: 15px;">
                    🤖 AI Medical Response
                </h4>
                <p style="color: #34495e; line-height: 1.6;">
                    {result['answer']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Answer generated successfully!")
    else:
        st.warning("⚠️ Please enter a medical question before submitting!")

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8); margin-top: 40px;">
    <p>❤️ Built with Streamlit & RAG | © 2025 Hack-A-Cure</p>
    <p style="font-size: 0.8rem;">
        🛡️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
    </p>
</div>
""", unsafe_allow_html=True)