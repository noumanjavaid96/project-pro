import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import json
from docx import Document
from PyPDF2 import PdfReader
import tempfile
import os
from typing import List, Dict
import time

# Page configuration
st.set_page_config(
    page_title="ABPI Code Compliance Checker",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f0fe;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'abpi_content' not in st.session_state:
    st.session_state.abpi_content = None
if 'doc_content' not in st.session_state:
    st.session_state.doc_content = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

def configure_gemini(api_key: str):
    """Configure Gemini with API key and return model"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return None

def process_file(file) -> str:
    """Process uploaded files and extract text content."""
    if file is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

            if file.type == "application/pdf":
                reader = PdfReader(tmp_file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text

            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(tmp_file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text

            elif file.type == "text/plain":
                return file.getvalue().decode("utf-8")

            else:
                st.error("Unsupported file format")
                return None

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
    finally:
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)

def analyze_document(model, abpi_content: str, doc_content: str) -> str:
    """Analyze document using Gemini Pro"""
    try:
        prompt = f"""As an ABPI Code compliance expert, analyze the following document:
        
{doc_content[:5000]}  # Limit document content to avoid token limits

Using these ABPI guidelines:
{abpi_content[:5000]}  # Limit ABPI content to avoid token limits

Provide a detailed analysis including:
1. Overall compliance assessment
2. Specific areas of concern (if any)
3. Recommendations for improvement
4. References to relevant ABPI Code sections

Format your response in clear sections with markdown formatting."""

        response = model.generate_content(prompt)
        full_text = ""
        for part in response.parts:  # Assuming 'parts' is the correct attribute
            if part.text:
                full_text += part.text

        return full_text if full_text else "No analysis generated"

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def get_chat_response(model, question: str, context: str) -> str:
    """Get response from Gemini with context"""
    try:
        # ... (rest of your code) ...

        # Construct the prompt with context
        prompt = f"""Based on this ABPI compliance analysis context:
{context[:2000]} 

Answer the following question:
{question}

Remember to reference specific sections of the ABPI Code where relevant."""

        response = chat_session.send_message(prompt)

        # Access and concatenate text from response parts
        full_text = ""
        for part in response.parts:
            if part.text:
                full_text += part.text

        return full_text if full_text else "Unable to generate response"

    except Exception as e:
        st.error(f"Error in chat: {str(e)}")
        return None

# Main interface
st.title("⚖️ ABPI Code Compliance Checker")
st.markdown("Upload documents and analyze compliance with ABPI Code.")

# API Key input in sidebar
with st.sidebar:
    st.header("Configuration")
    gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")
    if gemini_api_key:
        st.session_state.gemini_model = configure_gemini(gemini_api_key)

# Main content
if st.session_state.gemini_model:
    # File upload section
    st.header("Document Upload")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Upload ABPI Code Document", expanded=True):
            abpi_code_file = st.file_uploader(
                "Upload ABPI Code (PDF/DOCX/TXT)",
                type=["pdf", "docx", "txt"],
                key="abpi_code"
            )
            if abpi_code_file:
                st.session_state.abpi_content = process_file(abpi_code_file)
                if st.session_state.abpi_content:
                    st.success("✅ ABPI Code processed successfully!")

    with col2:
        with st.expander("Upload Document to Check", expanded=True):
            user_doc = st.file_uploader(
                "Upload Document (PDF/DOCX/TXT)",
                type=["pdf", "docx", "txt"],
                key="user_doc"
            )
            if user_doc:
                st.session_state.doc_content = process_file(user_doc)
                if st.session_state.doc_content:
                    st.success("✅ Document processed successfully!")

    # Analysis section
    if st.session_state.abpi_content and st.session_state.doc_content:
        if st.button("🔍 Start Compliance Analysis", use_container_width=True):
            with st.spinner("Analyzing documents for ABPI compliance..."):
                analysis_result = analyze_document(
                    st.session_state.gemini_model,
                    st.session_state.abpi_content,
                    st.session_state.doc_content
                )
                
                if analysis_result:
                    st.session_state.analysis_result = analysis_result
                    st.session_state.chat_history = []  # Reset chat history
                    
                    st.markdown("### 📊 Analysis Results")
                    st.markdown(analysis_result)

    # Chat interface
        if st.session_state.analysis_result:
            st.markdown("---")
            st.header("💬 Ask Questions About the Analysis")
            generation_config = {
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-002",
                generation_config=generation_config,
            )
            chat_session = model.start_chat(history=[])
        else:
            st.error("Gemini API key not found in environment variables.")


        # Initialize chat history if not already present
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        user_input = st.chat_input("Enter your question here...")
        if user_input:
            # Display user message in chat
            with st.chat_message("user"):
                st.markdown(user_input)

            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Get response from Gemini
            response = get_chat_response(
                st.session_state.gemini_model,
                user_input,
                st.session_state.analysis_result
            )

            # Display assistant response in chat
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            # No need to manually rerun with st.rerun() here, 
            # Streamlit's chat input handles updates automatically

else:
    st.warning("👋 Please enter your Gemini API key in the sidebar to begin.")
