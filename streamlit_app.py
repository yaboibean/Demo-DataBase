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
customer_need = st.text_input(
    "Enter the client's problem:",
    key="customer_need",
    help="Type your client's need and press Enter to search.",
    placeholder="Describe your client's problem here...",
)
# Add custom CSS to make the input box much larger
st.markdown('''
<style>
input#customer_need {
    height: 4em !important;
    font-size: 1.5em !important;
    padding: 1.8em 1.2em !important;
    border-radius: 1em !important;
}
</style>
''', unsafe_allow_html=True)

# Number of results
top_k = st.sidebar.slider("Number of top matches", 1, 10, 2)

# --- FILTERS/FACETED SEARCH ---
# Load the full DataFrame for filter options
try:
    df_full = pd.read_csv(SPREADSHEET_PATH)
except Exception as e:
    st.error(f"Could not load spreadsheet for filters: {e}")
    df_full = None

# Define filter columns (update as needed)
FILTER_COLS = [
    ("Industry", "Industry"),
    ("Date Uploaded", "Date Uploaded"),
    ("AI Capability", "Instalily AI Capabilities"),
    ("Benefit", "Benefit to Client"),
]

selected_filters = {}
if df_full is not None:
    st.sidebar.markdown("### Filter Demos")
    for label, col in FILTER_COLS:
        if col in df_full.columns:
            options = ["All"] + sorted(df_full[col].dropna().unique().tolist())
            selected = st.sidebar.selectbox(f"{label}", options, key=f"filter_{col}")
            selected_filters[col] = selected

# --- MAIN SEARCH AND MATCHING ---
if customer_need.strip():
    with st.spinner('🔎 The AI model is analyzing your request and searching for the best matches...'):
        try:
            matcher = OpenAIGPTMatcher(SPREADSHEET_PATH, MATCH_COLUMNS, openai_api_key)
            results = matcher.find_best_demos(customer_need, top_k=top_k)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # --- APPLY FILTERS TO RESULTS ---
    if df_full is not None and any(v != "All" for v in selected_filters.values()):
        filtered = []
        for res in results:
            demo = res.get('demo_info', {})
            match = True
            for col, val in selected_filters.items():
                if val != "All" and demo.get(col, None) != val:
                    match = False
                    break
            if match:
                filtered.append(res)
        results = filtered

    if not results:
        st.info("No relevant demos found.")
    else:
        st.subheader("")
        for res in results:
            demo = res.get('demo_info', {})
            # --- DEMO VIDEO PREVIEW ---
            video_url = demo.get('Demo Video Link') or demo.get('Video Link') or demo.get('Demo link')
            # Use a container with st.markdown for the card, and apply the card CSS class
            st.markdown(f"""
<div class='result-card'>

<span style='font-size:2.2em; font-weight:800'>{demo.get(COMPANY_COL, 'N/A')}</span>

[Demo Link: Click here]({res.get('demo_link')})

**Date Uploaded:** {demo.get('Date Uploaded', 'N/A')}

**⭐ Similarity Score:** {res.get('similarity_score', 'N/A'):.3f}

**Reason:** <span style='color:#00FFAA'>{res.get('explanation', 'N/A')}</span>

**Client Problem:** {demo.get('Client Problem', '')}

**Instalily AI Capabilities:** {demo.get('Instalily AI Capabilities', '')}

**Benefit to Client:** {demo.get('Benefit to Client', '')}

""", unsafe_allow_html=True)
            # Embed video if available and is a YouTube or mp4 link
            if video_url and ("youtube.com" in video_url or "youtu.be" in video_url):
                st.video(video_url)
            elif video_url and video_url.endswith(('.mp4', '.webm', '.mov')):
                st.video(video_url)
            elif video_url and video_url.startswith('http'):
                st.markdown(f"[Preview Video]({video_url})")
            st.markdown("</div>", unsafe_allow_html=True)

# --- CHATBOT/CONVERSATIONAL UI ---
# Simple chatbot: maintain chat history and allow follow-up/refinement
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Chatbot Assistant")
chat_input = st.sidebar.text_input("Ask the AI for help or suggestions:", key="chat_input", placeholder="E.g. Suggest search terms for insurance demos")
if chat_input.strip():
    # For demo: Use matcher to suggest search terms or clarify
    try:
        matcher = OpenAIGPTMatcher(SPREADSHEET_PATH, MATCH_COLUMNS, openai_api_key)
        response = matcher.suggest_search_terms(chat_input)
        st.session_state['chat_history'].append((chat_input, response))
    except Exception as e:
        st.session_state['chat_history'].append((chat_input, f"Error: {e}"))

if st.session_state['chat_history']:
    for user, bot in st.session_state['chat_history'][-5:]:
        st.sidebar.markdown(f"**You:** {user}")
        st.sidebar.markdown(f"**AI:** {bot}")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Instalily AI. Secure & ready for Streamlit Community Cloud.")
