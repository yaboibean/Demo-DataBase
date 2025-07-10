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
SPREADSHEET_PATH = "Copy of Master File Demos Database - Demos Database.csv"
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

# Make sure df_full is defined before chatbot logic
try:
    df_full = pd.read_csv(SPREADSHEET_PATH)
except Exception as e:
    df_full = None

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("### Filter Demos")
# Number of results
if 'top_k' not in st.session_state:
    st.session_state['top_k'] = 2
st.sidebar.slider("Number of top matches", 1, 10, st.session_state['top_k'], key="top_k")

# Only keep Industry and Date Uploaded filters
FILTER_COLS = [
    ("Industry", "Industry"),
]
selected_filters = {}
start_date = None
end_date = None
if df_full is not None:
    # Industry filter
    for label, col in FILTER_COLS:
        if col in df_full.columns:
            options = ["All"] + sorted(df_full[col].dropna().unique().tolist())
            selected = st.sidebar.selectbox(f"{label}", options, key=f"filter_{col}")
            selected_filters[col] = selected
    # Date Uploaded range filter
    if "Date Uploaded" in df_full.columns:
        min_date = pd.to_datetime(df_full["Date Uploaded"], errors='coerce').min()
        max_date = pd.to_datetime(df_full["Date Uploaded"], errors='coerce').max()
        start_date = st.sidebar.date_input("Start Date (Date Uploaded)", value=min_date.date() if pd.notnull(min_date) else None, key="start_date")
        end_date = st.sidebar.date_input("End Date (Date Uploaded)", value=max_date.date() if pd.notnull(max_date) else None, key="end_date")

# --- MAIN PAGE CHATBOT UI ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.markdown("""
<style>
.big-chatbot-card {
    background: linear-gradient(135deg, #232946 80%, #1E90FF 100%);
    border-radius: 1.5em;
    padding: 2.2em 1.2em 1.5em 1.2em;
    margin: 2.5em auto 2em auto;
    box-shadow: 0 4px 24px 0 rgba(30,144,255,0.13);
    border: 2px solid #22263a;
    max-width: 700px;
}
.big-chatbot-title {
    font-size: 2.1em;
    font-weight: 900;
    color: #FFD700;
    margin-bottom: 0.5em;
    display: flex;
    align-items: center;
    gap: 0.7em;
    letter-spacing: 1px;
    justify-content: center;
}
.big-chatbot-input input {
    font-size: 1.3em !important;
    padding: 1.1em 1.2em !important;
    border-radius: 1em !important;
    margin-bottom: 1.2em !important;
}
.big-chatbot-response {
    color: #fff;
    font-size: 1.18em;
    background: #232946;
    border-radius: 1em;
    padding: 1em 1.2em;
    margin-bottom: 0.7em;
    word-break: break-word;
}
</style>
<div class='big-chatbot-card'>
    <div class='big-chatbot-title'>🤖 InstaDemo Chatbot</div>
    <div style='font-size:1.1em; color:#fff; margin-bottom:1em; text-align:center;'>
        Enter a client issue (e.g. <b>"supply chain issues"</b>) or ask a question about B2B AI demos. The AI will find the most relevant demo(s) or answer your question using the database.
    </div>
""", unsafe_allow_html=True)

chat_input = st.text_input(
    "Describe a client issue or ask a question:",
    key="chat_input_main",
    placeholder="E.g. supply chain issues or What is a good demo for insurance fraud detection?",
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

if chat_input.strip():
    with st.spinner('AI is thinking...'):
        try:
            if df_full is not None:
                preview_cols = [col for col in df_full.columns if col not in (None, '')]
                preview_df = df_full[preview_cols].head(20)
                sheet_summary = preview_df.to_csv(index=False)
            else:
                sheet_summary = "(Spreadsheet data unavailable)"
            system_prompt = (
                "You are an expert B2B AI demo assistant for a company that matches client needs to AI demos. "
                "If the user input is a client problem or issue (e.g. 'supply chain issues'), find and recommend the most relevant demo(s) from the database, always including the demo link. "
                "If the user input is a general question, answer it using only the information in the demo database, and include demo links if relevant. "
                "If the answer is not in the data, say so. "
                "Never hallucinate or make up demos. "
                "Be concise, accurate, and helpful. "
                "Here is the demo database (CSV):\n" + sheet_summary
            )
            prompt = f"User input: {chat_input}"
            import openai
            openai.api_key = openai_api_key
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt},
                         {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            content = response.choices[0].message.content
            answer = content.strip() if content else "(No response from AI)"
            st.session_state['chat_history'].append((chat_input, answer))
        except Exception as e:
            st.session_state['chat_history'].append((chat_input, f"Error: {e}"))

if st.session_state['chat_history']:
    st.markdown("<div class='big-chatbot-card'>", unsafe_allow_html=True)
    st.markdown("<b style='font-size:1.2em;'>Recent Chat</b>", unsafe_allow_html=True)
    for user, bot in st.session_state['chat_history'][-5:]:
        st.markdown(f"<div class='big-chatbot-response'><b>You:</b> {user}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-chatbot-response'><b>AI:</b> {bot}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Developed by InstaLILY AI. Secure & ready for Streamlit Community Cloud.")
