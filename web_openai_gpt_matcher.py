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
# Only use the 'Client Problem' column for matching
match_columns = ["Client Problem"]

api_key = get_api_key()

def extract_company_names_and_indices(gpt_output):
    # Extract company names and their order from GPT output numbered list
    pattern = r"(\d+)\.\s*([^:]+):"
    return [(int(num), name.strip()) for num, name in re.findall(pattern, gpt_output)]

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
            # Use the correct method name for GPT-4o matching
            result = matcher.find_best_demos(customer_need, top_k=num_results)
        st.subheader("GPT-4o Top Matches:")
        # Display results in a readable format
        import json
        if isinstance(result, list):
            for match in result:
                info = match.get('demo_info', {})
                score = match.get('similarity_score', 0)
                rank = match.get('rank', '?')
                company = info.get('Name/Client', 'Unknown Company')
                st.markdown(f"### {rank}. {company}")
                st.markdown(f"**Similarity Score:** {score:.3f}")
                if 'Client Problem' in info and pd.notna(info['Client Problem']):
                    st.markdown(f"**Client Problem:** {info['Client Problem']}")
                video_link = info.get('Video Link')
                if video_link and pd.notna(video_link):
                    st.markdown(f"**Video Link:** [{video_link}]({video_link})")
                st.markdown("---")
        else:
            st.markdown(result)
