#!/usr/bin/env python3
"""
Streamlit Frontend for Real Estate Asset Management Assistant
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time
import re
import html

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Real Estate Asset Management Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Chat message styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background-color: #2c2c2c;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 90%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Classification badges */
    .classification-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    .relevant {
        background-color: #28a745;
        color: white;
    }
    
    .irrelevant {
        background-color: #dc3545;
        color: white;
    }
    
    .off-topic {
        background-color: #c1c142;
        color: white;
    }
    
    /* SQL query styling */
    .sql-query {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .status-info {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem 0;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit progress bar - Multiple approaches */
    [data-testid="stStatusWidget"] { display: none !important; }
    .css-1n76uvr.e1fqkh3o1 { display: none !important; }
    .stStatusWidget { display: none !important; }
    
    /* More aggressive selectors */
    div[data-testid="stStatusWidget"] { display: none !important; }
    .stStatusWidget, .stStatusWidget * { display: none !important; }
    
    /* Hide any element with status in class name */
    [class*="status"] { display: none !important; }
    [class*="Status"] { display: none !important; }
    
    /* Hide progress indicators */
    [class*="progress"] { display: none !important; }
    [class*="Progress"] { display: none !important; }
    
    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Additional Streamlit elements to hide */
    .stDeployButton { display: none !important; }
    .reportview-container .main .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Check if the backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def get_agent_info():
    """Get information about the agent"""
    try:
        response = requests.get(f"{BACKEND_URL}/agent/info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None

def send_chat_message(message):
    """Send a message to the backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message, "user_id": "streamlit_user"},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"response": f"Error: {response.status_code}", "error": "HTTP Error"}
    except requests.exceptions.RequestException as e:
        return {"response": f"Connection error: {str(e)}", "error": str(e)}

def display_classification_badge(classification):
    """Display a badge for the query classification"""
    if not classification:
        return ""
    
    badge_class = {
        "relevant_query": "relevant",
        "irrelevant_query": "irrelevant", 
        "other_subject": "other-subject"
    }.get(classification, "other-subject")
    
    badge_text = {
        "relevant_query": "Property Query",
        "irrelevant_query": "Missing Data", 
        "other-subject": "Off Topic"
    }.get(classification, "Unknown")
    
    return f'<span class="classification-badge {badge_class}">{badge_text}</span>'

def clean_and_format_response(response_text):
    """Clean and format the response text from the agent"""
    if not response_text:
        return ""
    
    # Decode HTML entities
    cleaned = html.unescape(response_text)
    
    # Remove HTML tags but preserve line breaks
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Clean up extra whitespace but preserve intentional line breaks
    # Replace multiple spaces with single space
    cleaned = re.sub(r' +', ' ', cleaned)
    
    # Replace multiple newlines with double newlines for paragraph separation
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    # Clean up leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Format paragraphs
    paragraphs = cleaned.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # Add proper spacing for bullet points
            if paragraph.startswith('•') or paragraph.startswith('-'):
                paragraph = f"  {paragraph}"
            formatted_paragraphs.append(paragraph)
    
    return '\n\n'.join(formatted_paragraphs)

def format_sql_query(sql_text):
    """Format SQL query for better readability"""
    if not sql_text:
        return ""
    
    # Clean the SQL query
    cleaned = html.unescape(sql_text)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Add syntax highlighting keywords
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'COUNT', 'SUM', 'AVG', 'DISTINCT', 'AS', 'AND', 'OR']
    for keyword in keywords:
        cleaned = re.sub(rf'\b{keyword}\b', f'<span style="color: #0066cc; font-weight: bold;">{keyword}</span>', cleaned, flags=re.IGNORECASE)
    
    return cleaned

def display_error_message(error_text):
    """Display error messages with proper formatting"""
    if not error_text:
        return ""
    
    # Clean the error message
    cleaned = html.unescape(error_text)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    return f'<div style="color: #d32f2f; background: #ffebee; padding: 1rem; border-radius: 8px; border-left: 4px solid #d32f2f;">⚠️ {cleaned}</div>'

def main():
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🏢 Real Estate Asset Management Assistant</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Ask questions about your property portfolio and get intelligent insights!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        # Check backend health
        backend_healthy, health_data = check_backend_health()
        
        if backend_healthy:
            st.markdown('<div class="status-success">✅ Backend Connected</div>', unsafe_allow_html=True)
            if health_data and health_data.get("agent_ready"):
                st.markdown('<div class="status-success">✅ Agent Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">❌ Agent Not Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Backend Disconnected</div>', unsafe_allow_html=True)
            st.info("Make sure the backend is running on http://localhost:8000")
        
        # Agent info with better formatting
        if backend_healthy:
            agent_info = get_agent_info()
            if agent_info:
                st.markdown("### 📋 Database Info")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Table", agent_info['database_info']['table_name'])
                with col2:
                    st.metric("Rows", f"{agent_info['database_info']['row_count']:,}")
                
                st.markdown("### ⚙️ Agent Config")
                config = agent_info['agent_config']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model", config['model'])
                with col2:
                    st.metric("Temperature", config['temperature'])
                st.metric("Max Iterations", config['max_iterations'])
        
        st.markdown("### 💡 Example Queries")
        with st.expander("Click to see examples"):
            st.markdown("""
            **🏠 Property-Related Queries:**
            - "What's the total profit for 2024?"
            - "Show me the top 5 properties by profit"
            - "Which entity has the highest revenue?"
            
            **❌ Irrelevant Property Queries:**
            - "What's the address of each property?"
            - "How many square feet is each building?"
            - "What's the property valuation?"
            
            **🌤️ Off-Topic Queries:**
            - "What's the weather like today?"
            - "Tell me a joke"
            """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown("### 💬 Chat")
        
        # Display chat history with improved styling
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Clean and format the response
                cleaned_response = clean_and_format_response(message["content"])
                
                # Start the assistant message container
                st.markdown('<div class="assistant-message">', unsafe_allow_html=True)
                st.markdown("**🏢 Assistant:**", unsafe_allow_html=True)
                
                # Display classification badge if available
                if "classification" in message and message["classification"]:
                    badge_html = display_classification_badge(message["classification"])
                    st.markdown(badge_html, unsafe_allow_html=True)
                
                # Display the response text
                st.markdown(f"<div style='color: white; margin: 10px 0;'>{cleaned_response}</div>", unsafe_allow_html=True)
                
                # Display SQL query if available
                if "sql_query" in message and message["sql_query"]:
                    st.markdown("**🔍 SQL Query:**", unsafe_allow_html=True)
                    st.code(message["sql_query"], language="sql")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with improved styling
    if backend_healthy:
        st.markdown("### 💭 Ask a Question")
        
        # Use form for better UX
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask about your property portfolio:",
                key="user_input",
                height=100,
                placeholder="e.g., What's the total profit for 2024?"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                submit_button = st.form_submit_button("🚀 Send", type="primary")
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
            
            if submit_button and user_input.strip():
                # Add user message to chat
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Show spinner while processing
                with st.spinner("🤖 Processing your property query..."):
                    # Send to backend
                    response = send_chat_message(user_input)
                    
                    # Check if there's an error
                    if "error" in response and response["error"]:
                        error_message = display_error_message(response["response"])
                        st.markdown(error_message, unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ Error: {response['response']}",
                            "classification": "error",
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # Clean and format the response
                        cleaned_response = clean_and_format_response(response["response"])
                        
                        # Add assistant response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": cleaned_response,  # Store cleaned version
                            "classification": response.get("query_classification"),
                            "sql_query": response.get("sql_query"),
                            "timestamp": datetime.now().isoformat()
                        })
                
                st.rerun()
            
            if clear_button:
                st.session_state.messages = []
                st.rerun()
    else:
        st.markdown('<div class="status-error">⚠️ Backend is not connected. Please start the backend server first.</div>', unsafe_allow_html=True)
        st.code("cd backend && python main.py", language="bash")
    
    # Modern footer
    st.markdown("""
    <div class="footer">
        Built with ❤️ using Streamlit, FastAPI, and LangGraph<br>
        Real Estate Asset Management Assistant
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 