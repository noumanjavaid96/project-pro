import streamlit as st
from openai import OpenAI
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
        .analysis-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #ddd;
        }
        .violation {
            background-color: #ffe6e6;
            border-left: 4px solid #ff4444;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .recommendation {
            background-color: #e6ffe6;
            border-left: 4px solid #44ff44;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .reference {
            background-color: #e6e6ff;
            border-left: 4px solid #4444ff;
            padding: 1rem;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def chunk_text(text: str, chunk_size: int = 200000) -> List[str]:
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

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

def analyze_chunk(client: OpenAI, chunk: str, abpi_code: str) -> Dict:
    """Analyze a chunk of text for compliance using GPT-4o."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                You are an expert in ABPI Code compliance analysis.
                Analyze the provided text chunk and return a JSON object with:
                {
                    "violations": ["list of identified violations"],
                    "recommendations": ["list of recommendations"],
                    "references": ["list of relevant ABPI Code references"],
                    "compliance_score": "percentage of compliance"
                }
                Be specific and detailed in your analysis.
                """},
                {"role": "user", "content": f"ABPI Code: {abpi_code[:1000]}...\n\nDocument chunk to analyze: {chunk}"}
            ],
            temperature=0.2,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error analyzing chunk: {str(e)}")
        return None

def merge_analyses(analyses: List[Dict]) -> Dict:
    """Merge multiple chunk analyses into a single result."""
    merged = {
        "violations": [],
        "recommendations": [],
        "references": [],
        "compliance_scores": []
    }
    
    for analysis in analyses:
        if analysis:
            merged["violations"].extend(analysis.get("violations", []))
            merged["recommendations"].extend(analysis.get("recommendations", []))
            merged["references"].extend(analysis.get("references", []))
            if "compliance_score" in analysis:
                merged["compliance_scores"].append(float(analysis["compliance_score"].rstrip('%')))
    
    # Remove duplicates while preserving order
    merged["violations"] = list(dict.fromkeys(merged["violations"]))
    merged["recommendations"] = list(dict.fromkeys(merged["recommendations"]))
    merged["references"] = list(dict.fromkeys(merged["references"]))
    
    # Calculate average compliance score
    if merged["compliance_scores"]:
        merged["overall_compliance"] = f"{sum(merged['compliance_scores']) / len(merged['compliance_scores']):.1f}%"
    else:
        merged["overall_compliance"] = "N/A"
    
    return merged

# Main interface
st.title("⚖️ ABPI Code Compliance Checker")
st.markdown("Upload documents and analyze compliance with ABPI Code using GPT-4o")

# API Key input
api_key = st.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    client = OpenAI(api_key=api_key)

    # File upload section
    st.subheader("Document Upload")
    col1, col2 = st.columns(2)
    
    with col1:
        abpi_code_file = st.file_uploader(
            "Upload ABPI Code (Source of Truth)",
            type=["pdf", "docx", "txt"],
            key="abpi_code"
        )
        if abpi_code_file:
            abpi_content = process_file(abpi_code_file)
            if abpi_content:
                st.success("ABPI Code processed successfully!")

    with col2:
        user_doc = st.file_uploader(
            "Upload Document to Check",
            type=["pdf", "docx", "txt"],
            key="user_doc"
        )
        if user_doc:
            doc_content = process_file(user_doc)
            if doc_content:
                st.success("Document processed successfully!")

    # Analysis section
    if abpi_content and doc_content:
        st.subheader("Compliance Analysis")
        
        if st.button("Analyze Compliance"):
            with st.spinner("Analyzing compliance..."):
                try:
                    # Split document into chunks
                    chunks = chunk_text(doc_content)
                    progress_bar = st.progress(0)
                    analyses = []
                    
                    # Analyze each chunk
                    for i, chunk in enumerate(chunks):
                        analysis = analyze_chunk(client, chunk, abpi_content)
                        if analysis:
                            analyses.append(analysis)
                        progress_bar.progress((i + 1) / len(chunks))
                        time.sleep(1)  # Prevent rate limiting
                    
                    # Merge results
                    final_analysis = merge_analyses(analyses)
                    
                    # Display overall compliance score
                    st.markdown(f"### Overall Compliance Score: {final_analysis['overall_compliance']}")
                    
                    # Create tabs for different sections
                    tabs = st.tabs(["Violations", "Recommendations", "References"])
                    
                    # Violations tab
                    with tabs[0]:
                        if final_analysis["violations"]:
                            for violation in final_analysis["violations"]:
                                st.markdown(f'<div class="violation">{violation}</div>', 
                                          unsafe_allow_html=True)
                        else:
                            st.success("No violations found")
                    
                    # Recommendations tab
                    with tabs[1]:
                        if final_analysis["recommendations"]:
                            for recommendation in final_analysis["recommendations"]:
                                st.markdown(f'<div class="recommendation">{recommendation}</div>', 
                                          unsafe_allow_html=True)
                        else:
                            st.info("No recommendations provided")
                    
                    # References tab
                    with tabs[2]:
                        if final_analysis["references"]:
                            for reference in final_analysis["references"]:
                                st.markdown(f'<div class="reference">{reference}</div>', 
                                          unsafe_allow_html=True)
                        else:
                            st.info("No specific references found")
                    
                    # Export results
                    if st.button("Export Analysis"):
                        st.download_button(
                            label="Download Analysis Report",
                            data=json.dumps(final_analysis, indent=2),
                            file_name="compliance_analysis.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                finally:
                    progress_bar.empty()

else:
    st.warning("Please enter your OpenAI API key to start.")

# Sidebar with information
with st.sidebar:
    st.markdown("""
        ### About
        This tool uses OpenAI's GPT-4o model to analyze
        documents for compliance with the ABPI Code.
        
        **Model Configuration:**
        - Model: gpt-4o
        - Max Tokens: 4096
        - Temperature: 0.2
        - Response Format: JSON
        
        ### Features
        - Document comparison
        - Compliance scoring
        - Detailed violation analysis
        - Actionable recommendations
        - Code references
        
        ### Supported Files
        - PDF (.pdf)
        - Word (.docx)
        - Text (.txt)
    """)

    if st.button("Clear All"):
        st.session_state.clear()
        st.experimental_rerun()
