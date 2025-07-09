import os
import streamlit as st
import pandas as pd
from demo_matcher import DemoMatcher
from openai_demo_matcher import OpenAIDemoMatcher
from openai_gpt_matcher import OpenAIGPTMatcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Demo Matcher",
    layout="wide"
)

# Title
st.title("AI-Powered Demo Matcher")

# File and columns
SPREADSHEET_PATH = os.getenv("DEMO_SPREADSHEET", "demo_data.csv")
MATCH_COLUMNS = ["Client Problem", "Instalily AI Capabilities", "Benefit to Client"]
VIDEO_LINK_COL = "Video Link"
COMPANY_COL = "Company Name"

# Load API key
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else os.getenv("OPENAI_API_KEY")

# Select matcher type
matcher_type = st.sidebar.selectbox(
    "Select Matching Engine",
    ("Local Embedding", "OpenAI Embedding", "OpenAI GPT-4o Reasoning")
)

# Input box
customer_need = st.text_area(
    "Enter the client's need/problem:",
    height=100
)

# Number of results
top_k = st.sidebar.slider("Number of top matches", 1, 10, 5)

# Run matcher
if st.button("Find Matches"):
    if not customer_need.strip():
        st.warning("Please enter a client need/problem.")
    else:
        try:
            if matcher_type == "Local Embedding":
                matcher = DemoMatcher(SPREADSHEET_PATH, MATCH_COLUMNS)
                results = matcher.find_similar_demos(customer_need, top_k=top_k)
            elif matcher_type == "OpenAI Embedding":
                if not openai_api_key:
                    st.error("OpenAI API key not found. Please set it in Streamlit secrets or .env.")
                    st.stop()
                matcher = OpenAIDemoMatcher(SPREADSHEET_PATH, MATCH_COLUMNS, openai_api_key)
                results = matcher.find_similar_demos(customer_need, top_k=top_k)
            else:  # OpenAI GPT-4o Reasoning
                if not openai_api_key:
                    st.error("OpenAI API key not found. Please set it in Streamlit secrets or .env.")
                    st.stop()
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
                company = demo.get(COMPANY_COL, 'N/A')
                video_link = demo.get(VIDEO_LINK_COL, '')
                similarity = res.get('similarity_score', None)
                explanation = res.get('explanation', None)
                st.markdown(f"**{res.get('rank', '')}. {company}**")
                if similarity is not None:
                    st.write(f"Similarity Score: {similarity:.2f}")
                if explanation:
                    st.write(f"Reason: {explanation}")
                if video_link:
                    st.markdown(f"[Watch Video]({video_link})")
                st.write(demo)
                st.markdown("---")

            # Bar chart for embedding-based matchers
            if matcher_type in ("Local Embedding", "OpenAI Embedding"):
                import numpy as np
                scores = [r['similarity_score'] for r in results]
                companies = [r['demo_info'].get(COMPANY_COL, 'N/A') for r in results]
                st.bar_chart(pd.DataFrame({'Similarity': scores}, index=companies))

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Instalily AI. Secure & ready for Streamlit Community Cloud.")
