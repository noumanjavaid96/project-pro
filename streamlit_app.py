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
import tiktoken

# Configure page
st.set_page_config(
    page_title="ABPI Code Compliance Checker",
    page_icon="⚖️",
    layout="wide"
)

def configure_gemini(api_key: str):
    """Configure Gemini with API key and return model"""
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )

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

def analyze_with_gemini(model, abpi_content: str, doc_content: str) -> str:
    """Analyze document using Gemini Pro"""
    prompt = f"""You are an expert in ABPI Code compliance analysis.

ABPI Code Content:
{abpi_content}

Document to Analyze:
{doc_content}

Please provide a detailed analysis including:
1. Overall compliance assessment
2. Specific areas of concern (if any)
3. Recommendations for improvement
4. References to relevant ABPI Code sections

Format your response in clear sections with markdown formatting."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini analysis: {str(e)}")
        return None

def analyze_with_openai(client: OpenAI, abpi_content: str, doc_content: str) -> str:
    """Analyze document using OpenAI (fallback method)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in ABPI Code compliance analysis."},
                {"role": "user", "content": f"""
                ABPI Code Content:
                {abpi_content[:4000]}...

                Document to Analyze:
                {doc_content[:4000]}...

                Provide a detailed analysis including:
                1. Overall compliance assessment
                2. Specific areas of concern
                3. Recommendations
                4. ABPI Code references
                """}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenAI analysis: {str(e)}")
        return None

# Main interface
st.title("⚖️ ABPI Code Compliance Checker")
st.markdown("Upload documents and analyze compliance with ABPI Code.")

# API Keys input
with st.expander("Configure API Keys"):
    gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")
    openai_api_key = st.text_input("Enter your OpenAI API key (fallback):", type="password")

if gemini_api_key:
    gemini_model = configure_gemini(api_key=gemini_api_key)
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)

    # File upload section
    st.subheader("Document Upload")
    col1, col2 = st.columns(2)

    with col1:
        abpi_code_file = st.file_uploader(
            "Upload ABPI Code Document (Source of Truth)",
            type=["pdf", "docx", "txt"],
            key="abpi_code"
        )
        if abpi_code_file:
            abpi_content = process_file(abpi_code_file)
            if abpi_content:
                st.success("ABPI Code document processed successfully!")

    with col2:
        user_doc = st.file_uploader(
            "Upload Document to Check for Compliance",
            type=["pdf", "docx", "txt"],
            key="user_doc"
        )
        if user_doc:
            doc_content = process_file(user_doc)
            if doc_content:
                st.success("Document to check processed successfully!")

    # Analysis section
    if abpi_code_file and user_doc and st.button("Start Analysis"):
        with st.spinner("Analyzing documents..."):
            try:
                # Try Gemini first
                analysis_result = analyze_with_gemini(gemini_model, abpi_content, doc_content)
                
                # Fallback to OpenAI if Gemini fails and OpenAI is configured
                if analysis_result is None and openai_api_key:
                    st.warning("Falling back to OpenAI analysis...")
                    analysis_result = analyze_with_openai(openai_client, abpi_content, doc_content)

                if analysis_result:
                    st.markdown("### Analysis Results")
                    st.markdown(analysis_result)

                    # Initialize chat interface
                    if 'conversation' not in st.session_state:
                        st.session_state['conversation'] = []
                        st.session_state['conversation'].append(analysis_result)

                    # Chat interface
                    st.subheader("Ask Questions About the Analysis")
                    user_input = st.text_input("Your question:", key="user_input")
                    
                    if user_input:
                        try:
                            chat_prompt = f"""Based on the previous analysis:
                            {analysis_result[:1000]}...
                            
                            Question: {user_input}
                            
                            Please provide a detailed answer."""

                            # Try Gemini first for chat
                            chat_response = gemini_model.generate_content(chat_prompt)
                            response_text = chat_response.text

                            st.markdown("**Question:**")
                            st.write(user_input)
                            st.markdown("**Answer:**")
                            st.write(response_text)

                            # Store in conversation history
                            st.session_state['conversation'].append({
                                "question": user_input,
                                "answer": response_text
                            })

                        except Exception as e:
                            st.error(f"Error in chat: {str(e)}")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
else:
    st.warning("Please enter your API keys to begin analysis.")
