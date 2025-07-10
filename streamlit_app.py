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

# Only keep Industry and Date Uploaded filters
FILTER_COLS = [
    ("Industry", "Industry"),
]

selected_filters = {}
start_date = None
end_date = None
if df_full is not None:
    st.sidebar.markdown("### Filter Demos")
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

# --- CHATBOT: SIDEBAR INPUT, SLIDE TO MAIN PANEL ON MESSAGE ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Sidebar chatbot input and chat history
show_sidebar_chat = not st.session_state['chat_history'] or not st.session_state.get('expand_chat', False)
if show_sidebar_chat:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Chatbot Assistant")
    chat_input = st.sidebar.text_input(
        "Ask the AI a question about B2B AI demos:",
        key="chat_input_sidebar",
        placeholder="E.g. What is a good demo for insurance fraud detection?"
    )
    if chat_input.strip():
        st.session_state['expand_chat'] = True
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
                    "Always answer the user's question directly and concisely first. "
                    "Then, provide additional relevant information from the demo database. "
                    "If you mention a demo, always include its link. "
                    "You have access to a database of past demos in CSV format. "
                    "For every user question, use only the information in the provided database to answer. "
                    "If the answer is not in the data, say so. "
                    "Be concise, accurate, and helpful. "
                    "Never hallucinate or make up demos. "
                    "If the user asks for a recommendation, suggest demos from the database that best match their question, and always provide the demo link. "
                    "If the user asks about a specific client, capability, or benefit, use the relevant fields from the database and provide the demo link if available. "
                    "If the user asks for a summary, provide a brief overview based on the data. "
                    "Here is the demo database (CSV):\n" + sheet_summary
                )
                prompt = f"User question: {chat_input}"
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

# --- MODERN DYNAMIC LAYOUT: MAIN AREA SPLIT ---
if st.session_state.get('expand_chat', False):
    # --- CENTERED CHATBOT PANEL, FULL WIDTH ---
    st.markdown('''
    <style>
    .centered-chat-transition {
        animation: slideInChat 0.6s cubic-bezier(.68,-0.55,.27,1.55) forwards;
        opacity: 0;
    }
    @keyframes slideInChat {
        from {
            transform: translateY(60px) scale(0.95);
            opacity: 0;
        }
        to {
            transform: translateY(0) scale(1);
            opacity: 1;
        }
    }
    .centered-chat-panel {
        background: #181c2b;
        border-radius: 1.2em;
        padding: 2em 2.5em 1em 2.5em;
        min-height: 500px;
        max-width: 800px;
        margin: 3em auto 2em auto;
        box-shadow: 0 2px 32px 0 rgba(30,144,255,0.18);
        border: 1px solid #22263a;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        width: 90vw;
    }
    .modern-chat-bubble-user {
        background: #23272f;
        color: #fff;
        border-radius: 1.2em 1.2em 0.3em 1.2em;
        padding: 0.7em 1.1em;
        margin-bottom: 0.3em;
        align-self: flex-end;
        max-width: 90%;
        animation: fadeIn 0.3s;
    }
    .modern-chat-bubble-bot {
        background: #ececf1;
        color: #222;
        border-radius: 1.2em 1.2em 1.2em 0.3em;
        padding: 0.7em 1.1em;
        margin-bottom: 0.3em;
        align-self: flex-start;
        max-width: 90%;
        animation: fadeIn 0.3s;
    }
    .modern-chat-label {
        font-size: 0.85em;
        color: #888;
        margin-bottom: 0.1em;
        margin-left: 0.2em;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    ''', unsafe_allow_html=True)
    st.markdown("<div class='centered-chat-panel centered-chat-transition'>", unsafe_allow_html=True)
    for user, bot in st.session_state['chat_history'][-8:]:
        st.markdown(f"<div class='modern-chat-label'>You</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='modern-chat-bubble-user'>{user}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='modern-chat-label'>AI</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='modern-chat-bubble-bot'>{bot}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    # --- TWO COLUMN LAYOUT: SEARCH + CHATBOT (SIDEBAR) ---
    col_search, col_chat = st.columns([2, 1], gap="large")
    with col_search:
        # --- MAIN SEARCH AND MATCHING ---
        if customer_need.strip():
            if not isinstance(openai_api_key, str) or not openai_api_key:
                st.error("OpenAI API key is missing or invalid. Please set your API key in Streamlit secrets or your .env file.")
                st.stop()
            with st.spinner('🔎 The AI model is analyzing your request and searching for the best matches...'):
                try:
                    matcher = OpenAIGPTMatcher(SPREADSHEET_PATH, MATCH_COLUMNS, openai_api_key)
                    results = matcher.find_best_demos(customer_need, top_k=top_k)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # --- APPLY FILTERS TO RESULTS ---
            if df_full is not None:
                filtered = []
                for res in results:
                    demo = res.get('demo_info', {})
                    match = True
                    for col, val in selected_filters.items():
                        if val != "All" and demo.get(col, None) != val:
                            match = False
                            break
                    if match and start_date and end_date and "Date Uploaded" in demo:
                        try:
                            demo_date = pd.to_datetime(demo["Date Uploaded"], errors='coerce').date()
                            if demo_date < start_date or demo_date > end_date:
                                match = False
                        except Exception:
                            match = False
                    if match:
                        filtered.append(res)
                results = filtered
            if not results:
                st.info("No relevant demos found.")
            else:
                st.subheader("")
                for res in results:
                    demo = res.get('demo_info', {})
                    video_url = demo.get('Demo Video Link') or demo.get('Video Link') or demo.get('Demo link')
                    st.markdown(f"""
<div class='result-card'>
<span style='font-size:2.2em; font-weight:800'>{demo.get(COMPANY_COL, 'N/A')}</span>
[Demo Link: Click here]({res.get('demo_link')})
**Date Uploaded:** {demo.get('Date Uploaded', 'N/A')}
**⭐ Similarity Score:** {res.get('similarity_score', 'N/A'):.3f}
**Reason:** <span style='color:#00FFAA'>{res.get('explanation', 'N/A')}</span>
**Client Problem:** {demo.get('Client Problem', '')}
**InstaLILY AI Capabilities:** {demo.get('InstaLILY AI Capabilities', '')}
**Benefit to Client:** {demo.get('Benefit to Client', '')}
""", unsafe_allow_html=True)
                    if video_url and ("youtube.com" in video_url or "youtu.be" in video_url):
                        st.video(video_url)
                    elif video_url and video_url.endswith(('.mp4', '.webm', '.mov')):
                        st.video(video_url)
                    elif video_url and video_url.startswith('http'):
                        st.markdown(f"[Preview Video]({video_url})")
                    st.markdown("</div>", unsafe_allow_html=True)
    with col_chat:
        # Sidebar chatbot input and chat history
        if show_sidebar_chat:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🤖 Chatbot Assistant")
            chat_input = st.sidebar.text_input(
                "Ask the AI a question about B2B AI demos:",
                key="chat_input_sidebar",
                placeholder="E.g. What is a good demo for insurance fraud detection?"
            )
            if chat_input.strip():
                st.session_state['expand_chat'] = True
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
                            "Always answer the user's question directly and concisely first. "
                            "Then, provide additional relevant information from the demo database. "
                            "If you mention a demo, always include its link. "
                            "You have access to a database of past demos in CSV format. "
                            "For every user question, use only the information in the provided database to answer. "
                            "If the answer is not in the data, say so. "
                            "Be concise, accurate, and helpful. "
                            "Never hallucinate or make up demos. "
                            "If the user asks for a recommendation, suggest demos from the database that best match their question, and always provide the demo link. "
                            "If the user asks about a specific client, capability, or benefit, use the relevant fields from the database and provide the demo link if available. "
                            "If the user asks for a summary, provide a brief overview based on the data. "
                            "Here is the demo database (CSV):\n" + sheet_summary
                        )
                        prompt = f"User question: {chat_input}"
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
