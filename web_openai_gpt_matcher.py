import streamlit as st
import pandas as pd
from openai_gpt_matcher import OpenAIGPTMatcher
import os
from dotenv import load_dotenv
import re

st.set_page_config(page_title="AI Demo Matcher (GPT)", page_icon="🤖", layout="centered")
st.title("🤖 AI Demo Matcher (GPT Reasoning)")
st.write("This version uses GPT-4o to reason about the best matches, not cosine similarity.")

# Load .env if present
load_dotenv()
def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "")
    return key

csv_path = "Copy of Master File Demos Database - Demos Database.csv"
match_columns = ["Client Problem", "Instalily AI Capabilities", "Benefit to Client"]

api_key = get_api_key()

def extract_company_names(gpt_output):
    # Extract company names from GPT output numbered list
    pattern = r"\d+\.\s*([^:]+):"
    return re.findall(pattern, gpt_output)

customer_need = st.text_area(
    "Describe the current client's problem/need:",
    placeholder="e.g. We need an AI system to help with customer support and handle frequently asked questions",
    height=100
)

num_results = st.slider("Number of top matches to show", 1, 10, 3)

if st.button("Find Best Matches (GPT Reasoning)"):
    if not api_key or not api_key.startswith("sk-"):
        st.warning("Please set your OpenAI API key in the .env file as OPENAI_API_KEY=sk-...")
    elif not customer_need.strip():
        st.warning("Please enter a client problem/need.")
    else:
        with st.spinner("GPT-4o is reasoning about the best matches..."):
            matcher = OpenAIGPTMatcher(csv_path, match_columns, api_key)
            result = matcher.find_best_match(customer_need, top_k=num_results)
        st.subheader("GPT-4o Top Matches:")
        st.markdown(result)
        # Show video link for only the top matches
        company_names = extract_company_names(result)
        df = pd.read_csv(csv_path)
        for company in company_names:
            row = df[df['Name/Client'].astype(str).str.strip() == company.strip()]
            if not row.empty and 'Video Link' in row.columns:
                video_link = row.iloc[0]['Video Link']
                if pd.notna(video_link) and str(video_link).strip():
                    st.markdown(f"**Video Link for {company}:** [{video_link}]({video_link})")
