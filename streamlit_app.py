import os
import streamlit as st
import pandas as pd
from demo_matcher import DemoMatcher
from openai_demo_matcher import OpenAIDemoMatcher
from openai_gpt_matcher import OpenAIGPTMatcher
from dotenv import load_dotenv, find_dotenv

# Try to find and load the .env file from anywhere in the project
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

# Debug: Show the loaded API key (masked)
def mask_key(key):
    if not key or len(key) < 8:
        return key
    return key[:4] + '...' + key[-4:]

# Set page config
st.set_page_config(
    page_title="AI Demo Matcher",
    layout="wide"
)

# Title
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #FFF;
        margin-bottom: 0.5em;
        letter-spacing: 1px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2em;
    }
    .result-card {
        background: #181c2b;
        border-radius: 1.2em;
        padding: 1.5em 2em;
        margin-bottom: 2em;
        box-shadow: 0 2px 16px 0 rgba(30,144,255,0.08);
        border: 1px solid #22263a;
    }
    .score {
        color: #FFD700;
        font-size: 1.1em;
        font-weight: bold;
    }
    .reason {
        color: #00FFAA;
        font-size: 1em;
        margin-bottom: 0.5em;
    }
    .demo-link {
        color: #1E90FF;
        font-size: 1em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .field-label {
        color: #aaa;
        font-weight: 600;
    }
    </style>
    <div class='main-title'>InstaDemo Search</div>
    <div class='subtitle'>Find the best AI demo for your client's needs</div>
""", unsafe_allow_html=True)

# File and columns - Use the actual CSV file and correct column names
SPREADSHEET_PATH = os.getenv("DEMO_SPREADSHEET", "Copy of Master File Demos Database - Demos Database.csv")
MATCH_COLUMNS = ["Client Problem", "Instalily AI Capabilities", "Benefit to Client"]
VIDEO_LINK_COL = "Demo link"  # Updated to match actual column name
COMPANY_COL = "Name/Client"   # Updated to match actual column name

# Load API key robustly (strip whitespace just in case)
def get_openai_key():
    # Try Streamlit secrets first (for cloud), then .env (for local)
    try:
        if "openai_api_key" in st.secrets:
            return st.secrets["openai_api_key"].strip()
    except:
        pass
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    return None

openai_api_key = get_openai_key()

# Debug: Show the loaded API key (masked) in sidebar for troubleshooting
with st.sidebar:
    st.markdown("**Debug Info:**")
    st.write("OPENAI_API_KEY loaded:", mask_key(openai_api_key))
    st.write(".env path used:", dotenv_path)
    # Show a preview of the loaded spreadsheet for verification
    try:
        df_preview = pd.read_csv(SPREADSHEET_PATH)
        st.write("**Spreadsheet Preview:**")
        st.dataframe(df_preview.head(5))
    except Exception as e:
        st.error(f"Could not load spreadsheet: {e}")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets (for cloud) or in a .env file (for local use). The variable name must be 'OPENAI_API_KEY'.")
    st.stop()

# Input box
customer_need = st.text_area(
    "Enter the client's problem:",
    height=100,
    key="customer_need",
    on_change=None,
    help="Type your client's need and press Enter to search."
)

# Number of results
top_k = st.sidebar.slider("Number of top matches", 1, 10, 2)

# Run matcher automatically when input changes (simulate 'Enter' to submit)
if customer_need.strip():
    with st.spinner('🔎 The AI model is analyzing your request and searching for the best matches...'):
        try:
            matcher = OpenAIGPTMatcher(SPREADSHEET_PATH, MATCH_COLUMNS, openai_api_key)
            results = matcher.find_best_demos(customer_need, top_k=top_k)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    if not results:
        st.info("No relevant demos found.")
    else:
        st.subheader("")
        for res in results:
            demo = res.get('demo_info', {})
            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='margin-bottom:0.2em'>{demo.get(COMPANY_COL, 'N/A')}</h3>", unsafe_allow_html=True)
            st.markdown(f"<span class='field-label'>Date Uploaded:</span> {demo.get('Date Uploaded', 'N/A')}", unsafe_allow_html=True)
            st.markdown(f"<span class='score'>⭐ Similarity Score: {res.get('similarity_score', 'N/A'):.3f}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='reason'><b>Reason:</b> {res.get('explanation', 'N/A')}</div>", unsafe_allow_html=True)
            demo_link = res.get('demo_link')
            if demo_link:
                st.markdown(f"<div class='demo-link'><b>Demo Link:</b> <a href='{demo_link}' target='_blank'>Click here</a></div>", unsafe_allow_html=True)
            st.markdown(f"<span class='field-label'>Client Problem:</span> {demo.get('Client Problem', '')}", unsafe_allow_html=True)
            st.markdown(f"<span class='field-label'>Instalily AI Capabilities:</span> {demo.get('Instalily AI Capabilities', '')}", unsafe_allow_html=True)
            st.markdown(f"<span class='field-label'>Benefit to Client:</span> {demo.get('Benefit to Client', '')}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Instalily AI. Secure & ready for Streamlit Community Cloud.")
