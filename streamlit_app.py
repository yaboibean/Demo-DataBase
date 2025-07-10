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
st.title("AI-Powered Demo Matcher")

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
    "Enter the client's need/problem:",
    height=100
)

# Number of results
top_k = st.sidebar.slider("Number of top matches", 1, 10, 2)

# Run matcher
if st.button("Find Matches"):
    if not customer_need.strip():
        st.warning("Please enter a client need/problem.")
    else:
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
            st.subheader("Top Matches:")
            for res in results:
                demo = res.get('demo_info', {})
                # Only show the requested fields, word-for-word from the CSV
                fields_to_show = [
                    (COMPANY_COL, 'Company Name'),
                    ('Date Uploaded', 'Date Uploaded'),
                    ('Client Problem', 'Client Problem'),
                    ('Instalily AI Capabilities', 'Instalily AI Capabilities'),
                    ('Benefit to Client', 'Benefit to Client'),
                    (VIDEO_LINK_COL, 'Demo Link'),
                ]
                st.markdown(f"**{res.get('rank', '')}. {demo.get(COMPANY_COL, 'N/A')}**")
                for col, label in fields_to_show:
                    value = demo.get(col, '')
                    if col == VIDEO_LINK_COL and value:
                        st.markdown(f"**{label}:** [Link]({value})")
                    elif value:
                        st.write(f"**{label}:** {value}")
                st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Instalily AI. Secure & ready for Streamlit Community Cloud.")
