import streamlit as st
import pandas as pd
from openai_demo_matcher import OpenAIDemoMatcher
import os
from dotenv import load_dotenv
import numpy as np

st.set_page_config(page_title="AI Demo Matcher (OpenAI)", page_icon="🤖", layout="centered")
st.title("🤖 AI Demo Matcher (OpenAI)")
st.write("Enter your client problem/need below to find the most similar past demos from your database.")

# Load .env if present
load_dotenv()
def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "")
    return key

csv_path = "Copy of Master File Demos Database - Demos Database.csv"
match_columns = ["Client Problem", "Instalily AI Capabilities", "Benefit to Client"]

api_key = get_api_key()

customer_need = st.text_area(
    "Describe the current client's problem/need:",
    placeholder="e.g. We need an AI system to help with customer support and handle frequently asked questions",
    height=100
)

num_results = st.slider("Number of top matches to show", 1, 10, 3)

if st.button("Find Similar Demos (OpenAI)"):
    if not api_key or not api_key.startswith("sk-"):
        st.warning("Please set your OpenAI API key in the .env file as OPENAI_API_KEY=sk-...")
    elif not customer_need.strip():
        st.warning("Please enter a client problem/need.")
    else:
        with st.spinner("Analyzing and searching for similar demos using OpenAI embeddings..."):
            matcher = OpenAIDemoMatcher(csv_path, match_columns, api_key)
            # Get all similarities for graphing
            need_embedding = matcher._embed_texts([customer_need])[0].reshape(1, -1)
            similarities = matcher.demos_df.copy()
            similarities["Cosine Similarity"] = matcher.demo_embeddings.dot(need_embedding.T).flatten() / (
                (np.linalg.norm(matcher.demo_embeddings, axis=1) * np.linalg.norm(need_embedding))
            )
            # Print all similarities for debugging
            print("Cosine similarities for all rows:", similarities["Cosine Similarity"].values)
            # Show bar chart
            st.subheader("Cosine Similarity for All Demos")
            st.bar_chart(similarities[["Name/Client", "Cosine Similarity"]].set_index("Name/Client"))
            # Show top matches as before
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
                for col in match_columns:
                    if col in info and pd.notna(info[col]):
                        st.markdown(f"**{col}:** {info[col]}")
                demo_link = info.get('Portal Demo Link') or info.get('Onedrive Demo Link')
                if demo_link and pd.notna(demo_link):
                    st.markdown(f"**Demo Link:** [{demo_link}]({demo_link})")
                st.markdown("---")
