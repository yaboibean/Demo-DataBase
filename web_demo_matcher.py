import streamlit as st
from demo_matcher import DemoMatcher
import pandas as pd

st.set_page_config(page_title="AI Demo Matcher", page_icon="🤖", layout="centered")
st.title("🤖 AI Demo Matcher")
st.write("Enter a client problem/need below to find the most similar past demos from your database.")

# Load matcher (cache to avoid reloading on every input)
@st.cache_resource
def get_matcher():
    return DemoMatcher(
        spreadsheet_path="Copy of Master File Demos Database - Demos Database.csv",
        match_columns=["Client Problem"]  # Only use Client Problem for matching
    )
matcher = get_matcher()

# User input
customer_need = st.text_area(
    "Describe the current client's problem/need:",
    placeholder="e.g. We need an AI system to help with customer support and handle frequently asked questions",
    height=100
)

num_results = st.slider("Number of top matches to show", 1, 10, 2)

if st.button("Find Similar Demos"):
    if not customer_need.strip():
        st.warning("Please enter a client problem/need.")
    else:
        with st.spinner("Analyzing and searching for similar demos..."):
            results = matcher.find_similar_demos(customer_need, top_k=num_results)
        if not results:
            st.info("No similar demos found.")
        else:
            st.subheader("Top Matches:")
            for demo in results:
                info = demo['demo_info']
                score = demo['similarity_score']
                rank = demo['rank']
                company = info.get('Name/Client', 'Unknown Company')
                # Generous scoring description
                if score >= 0.7:
                    score_desc = "Excellent match"
                elif score >= 0.5:
                    score_desc = "Strong match"
                elif score >= 0.3:
                    score_desc = "Possible match"
                else:
                    score_desc = "Somewhat related"
                st.markdown(f"### {rank}. {company}")
                st.markdown(f"**Similarity Score:** {score:.3f} ({score_desc})")
                for col in matcher.match_columns:
                    if col in info and pd.notna(info[col]):
                        st.markdown(f"**{col}:** {info[col]}")
                # Show demo link if available
                demo_link = info.get('Portal Demo Link') or info.get('Onedrive Demo Link')
                if demo_link and pd.notna(demo_link):
                    st.markdown(f"**Demo Link:** [{demo_link}]({demo_link})")
                st.markdown("---")
