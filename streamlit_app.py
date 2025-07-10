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

# --- MAIN SEARCH AND MATCHING ---
if customer_need.strip():
    if not isinstance(openai_api_key, str) or not openai_api_key:
        st.error("OpenAI API key is missing or invalid. Please set your API key in Streamlit secrets or your .env file.")
        st.stop()
    with st.spinner('🔎 The AI model is analyzing your request and searching for the best matches...'):
        try:
            # Use CSV file directly, not Google Sheets
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
            # Industry filter
            for col, val in selected_filters.items():
                if val != "All" and demo.get(col, None) != val:
                    match = False
                    break
            # Date Uploaded filter
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

# --- ENHANCED CHATBOT/CONVERSATIONAL UI ---
# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Add custom CSS for chatbot
st.markdown("""
<style>
.chatbot-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border-radius: 1.5em;
    padding: 1.5em;
    margin: 2em 0;
    box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.chatbot-title {
    color: #ffffff;
    font-size: 1.8em;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1em;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}
.chat-message {
    margin: 1em 0;
    padding: 1em 1.2em;
    border-radius: 1.2em;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 0.3em;
}
.bot-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    margin-right: auto;
    border-bottom-left-radius: 0.3em;
}
.chat-input-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 1em;
    padding: 1em;
    margin-top: 1em;
}
.chat-examples {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.8em;
    padding: 1em;
    margin: 1em 0;
    border-left: 4px solid #4CAF50;
}
.example-questions {
    color: #e0e0e0;
    font-size: 0.9em;
    line-height: 1.6;
}
.clear-chat-btn {
    background: linear-gradient(45deg, #ff6b6b, #ee5a52);
    color: white;
    border: none;
    padding: 0.5em 1em;
    border-radius: 2em;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 1em;
}
.clear-chat-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Create main chatbot section
st.markdown("""
<div class='chatbot-container'>
    <div class='chatbot-title'>🤖 AI Demo Assistant</div>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class='chat-examples'>
        <div class='example-questions'>
            <strong>💡 Try asking:</strong><br>
            • "What demos work best for healthcare companies?"<br>
            • "Show me AI solutions for fraud detection?"<br>
            • "Which clients have used automation for sales?"<br>
            • "What are the top benefits mentioned in demos?"
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state['chat_history'] = []
        st.rerun()

# Chat input with enhanced styling
chat_input = st.text_input(
    "💬 Ask me anything about our B2B AI demos:",
    key="chat_input",
    placeholder="Type your question here and press Enter...",
    help="Ask questions about demos, clients, industries, or AI capabilities"
)

# Enhanced chat input CSS
st.markdown("""
<style>
div[data-testid="stTextInput"] > div > div > input {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 2em;
    padding: 1em 1.5em;
    font-size: 1.1em;
    transition: all 0.3s ease;
}
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #4CAF50;
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
    transform: translateY(-2px);
}
div[data-testid="stTextInput"] > div > div > input::placeholder {
    color: rgba(255, 255, 255, 0.7);
}
</style>
""", unsafe_allow_html=True)

# Process chat input
if chat_input.strip():
    if chat_input.strip().endswith('?'):
        with st.spinner('🤔 AI is thinking...'):
            try:
                # Prepare a summary of the spreadsheet for the prompt
                if df_full is not None:
                    preview_cols = [col for col in df_full.columns if col not in (None, '')]
                    preview_df = df_full[preview_cols].head(20)
                    sheet_summary = preview_df.to_csv(index=False)
                else:
                    sheet_summary = "(Spreadsheet data unavailable)"
                
                system_prompt = (
                    "You are an expert B2B AI demo assistant for Instalily AI. "
                    "You have access to a database of past demos in CSV format. "
                    "Provide helpful, accurate answers based only on the demo database. "
                    "Be concise but informative. Use emojis appropriately to make responses engaging. "
                    "If the answer isn't in the data, say so politely. "
                    "Focus on being helpful for sales teams looking for the right demo to show prospects. "
                    f"Demo database:\n{sheet_summary}"
                )
                
                prompt = f"User question: {chat_input}"
                
                # Use the same OpenAI API as the main model
                import openai
                openai.api_key = openai_api_key
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                content = response.choices[0].message.content
                answer = content.strip() if content else "🤖 Sorry, I couldn't generate a response."
                st.session_state['chat_history'].append((chat_input, answer))
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.session_state['chat_history'].append((chat_input, error_msg))
    else:
        st.session_state['chat_history'].append((
            chat_input, 
            "❓ Please end your question with a '?' so I know you're asking me something!"
        ))

# Display chat history with enhanced styling
if st.session_state['chat_history']:
    st.markdown("### 💬 Conversation History")
    
    # Show recent messages (last 6 for better UX)
    recent_messages = st.session_state['chat_history'][-6:]
    
    for i, (user_msg, bot_msg) in enumerate(recent_messages):
        # User message
        st.markdown(f"""
        <div class='chat-message user-message'>
            <strong>👤 You:</strong> {user_msg}
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f"""
        <div class='chat-message bot-message'>
            <strong>🤖 AI Assistant:</strong> {bot_msg}
        </div>
        """, unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 Chat Features
- **Smart Context**: AI knows about all demos in the database
- **Industry Insights**: Ask about specific sectors
- **Demo Recommendations**: Get tailored suggestions
- **Real Data**: Answers based on actual demo history
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Developed by Instalily AI**")
st.sidebar.markdown("✅ Secure & Cloud-Ready")
