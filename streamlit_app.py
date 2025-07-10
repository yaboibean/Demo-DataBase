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
        width: 100%;
        display: block;
        overflow-x: auto;
        min-width: 0;
        min-height: 0;
        word-break: break-word;
    }
    .send-btn {
        background: #22263a;
        border: none;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        margin-left: 0.5em;
        margin-top: 0.5em;
        transition: background 0.2s;
    }
    .send-btn:hover {
        background: #1E90FF;
    }
    .send-arrow {
        width: 28px;
        height: 28px;
        display: block;
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
    .input-row {
        display: flex;
        align-items: flex-end;
        gap: 0.5em;
        margin-bottom: 1.5em;
    }
    .input-container {
        position: relative;
        width: 100%;
        margin-bottom: 1.5em;
    }
    .send-btn-abs {
        position: absolute;
        right: 12px;
        bottom: 12px;
        background: #888;
        border: none;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: not-allowed;
        transition: background 0.2s;
        padding: 0;
        z-index: 2;
    }
    .send-btn-abs.enabled {
        background: #22263a;
        cursor: pointer;
    }
    .send-btn-abs.enabled:hover {
        background: #1E90FF;
    }
    .send-arrow-up {
        width: 28px;
        height: 28px;
        display: block;
    }
    .send-arrow-up.disabled path {
        stroke: #ccc;
    }
    .send-arrow-up.enabled path {
        stroke: #22263a;
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

# Input box and custom send button
from streamlit.components.v1 import html

with st.form(key="search_form", clear_on_submit=False):
    # Container for relative positioning
    st.markdown('''
    <style>
    .input-container {
        position: relative;
        width: 100%;
        margin-bottom: 1.5em;
    }
    .send-btn-abs {
        position: absolute;
        right: 12px;
        bottom: 12px;
        background: #888;
        border: none;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: not-allowed;
        transition: background 0.2s;
        padding: 0;
        z-index: 2;
    }
    .send-btn-abs.enabled {
        background: #22263a;
        cursor: pointer;
    }
    .send-btn-abs.enabled:hover {
        background: #1E90FF;
    }
    .send-arrow-up {
        width: 28px;
        height: 28px;
        display: block;
    }
    .send-arrow-up.disabled path {
        stroke: #ccc;
    }
    .send-arrow-up.enabled path {
        stroke: #22263a;
    }
    </style>
    <div class="input-container">
    ''', unsafe_allow_html=True)
    customer_need = st.text_area(
        "Enter the client's problem:",
        height=100,
        key="customer_need",
        help="Type your client's need and press Enter or click the arrow to search."
    )
    # Determine if button should be enabled
    btn_enabled = bool(customer_need.strip())
    btn_class = "send-btn-abs enabled" if btn_enabled else "send-btn-abs"
    arrow_class = "send-arrow-up enabled" if btn_enabled else "send-arrow-up disabled"
    btn_disabled = "" if btn_enabled else "disabled"
    # Upward arrow SVG
    arrow_svg = f'''
    <svg class="{arrow_class}" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="48" fill="white"/>
        <path d="M50 70 L50 30 M50 30 L35 45 M50 30 L65 45" stroke="#22263a" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    '''
    # Button HTML
    html(f'''
    <button type="submit" class="{btn_class}" {'disabled' if not btn_enabled else ''}>
        {arrow_svg}
    </button>
    </div>
    ''', height=80)
    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button(label="", help="Send")

# Number of results
top_k = st.sidebar.slider("Number of top matches", 1, 10, 2)

# Only show results if user has started typing and submitted
if customer_need.strip() and submitted:
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
