import streamlit as st
import pandas as pd
from demo_matcher import DemoMatcher
import os

# Set page config
st.set_page_config(
    page_title="Demo Database Matcher",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'matcher' not in st.session_state:
    st.session_state.matcher = None

# Title and description
st.title("🎯 AI Demo Database Matcher")
st.markdown("""
This tool helps you find the most similar past demos based on current customer needs.
Upload your demo database spreadsheet and search for the best matching demos.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Demo Database")
    
    uploaded_file = st.file_uploader(
        "Upload your spreadsheet",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your demo database"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load into matcher
        if st.session_state.matcher is None:
            st.session_state.matcher = DemoMatcher()
        
        try:
            st.session_state.matcher.load_spreadsheet(temp_file)
            st.success(f"✅ Loaded {len(st.session_state.matcher.demos_df)} demos")
            
            # Show column info
            st.subheader("📊 Database Info")
            st.write(f"**Columns:** {list(st.session_state.matcher.demos_df.columns)}")
            st.write(f"**Total Demos:** {len(st.session_state.matcher.demos_df)}")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Option to use sample data
    st.header("📝 Sample Data")
    if st.button("Use Sample Database"):
        if st.session_state.matcher is None:
            st.session_state.matcher = DemoMatcher()
        
        # Create and load sample data
        sample_file = "sample_demo_database.csv"
        st.session_state.matcher.create_sample_data(sample_file)
        st.session_state.matcher.load_spreadsheet(sample_file)
        st.success("✅ Sample database loaded!")
        st.rerun()

# Main interface
if st.session_state.matcher is not None and st.session_state.matcher.demos_df is not None:
    
    # Search interface
    st.header("🔍 Find Similar Demos")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        customer_need = st.text_area(
            "Describe the current customer's problem/need:",
            height=100,
            placeholder="e.g., We need an AI system to automate customer support and handle frequently asked questions for our e-commerce platform..."
        )
    
    with col2:
        st.write("**Settings:**")
        top_k = st.slider("Number of matches", 1, 10, 5)
        
        search_button = st.button("🔍 Search Similar Demos", type="primary")
    
    # Search results
    if search_button and customer_need:
        with st.spinner("Searching for similar demos..."):
            similar_demos = st.session_state.matcher.find_similar_demos(customer_need, top_k)
        
        if similar_demos:
            st.header("📊 Search Results")
            
            # Display results
            for demo in similar_demos:
                score = demo['similarity_score']
                info = demo['demo_info']
                rank = demo['rank']
                
                # Color coding based on similarity score
                if score > 0.7:
                    color = "🟢"
                elif score > 0.5:
                    color = "🟡"
                else:
                    color = "🔴"
                
                with st.expander(f"{color} Rank {rank} - Similarity: {score:.3f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'Company Name' in info and pd.notna(info['Company Name']):
                            st.write(f"**Company:** {info['Company Name']}")
                        if 'Industry' in info and pd.notna(info['Industry']):
                            st.write(f"**Industry:** {info['Industry']}")
                        if 'Demo Type' in info and pd.notna(info['Demo Type']):
                            st.write(f"**Demo Type:** {info['Demo Type']}")
                        if 'Success Rate' in info and pd.notna(info['Success Rate']):
                            st.write(f"**Success Rate:** {info['Success Rate']}")
                        if 'Date' in info and pd.notna(info['Date']):
                            st.write(f"**Date:** {info['Date']}")
                    
                    with col2:
                        if 'Problem/Need' in info and pd.notna(info['Problem/Need']):
                            st.write(f"**Problem:** {info['Problem/Need']}")
                        if 'Solution Provided' in info and pd.notna(info['Solution Provided']):
                            st.write(f"**Solution:** {info['Solution Provided']}")
                        if 'Demo Link/File' in info and pd.notna(info['Demo Link/File']):
                            st.write(f"**Demo Link:** {info['Demo Link/File']}")
            
            # Export results
            st.header("📥 Export Results")
            
            # Create results dataframe
            results_df = pd.DataFrame([
                {
                    'Rank': demo['rank'],
                    'Similarity Score': demo['similarity_score'],
                    **demo['demo_info']
                }
                for demo in similar_demos
            ])
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"similar_demos_results.csv",
                mime="text/csv"
            )
            
            # Show detailed analysis
            with st.expander("📋 Detailed Analysis"):
                analysis = st.session_state.matcher.get_detailed_analysis(customer_need, top_k)
                st.text(analysis)
        
        else:
            st.warning("No similar demos found.")
    
    # Show current database
    st.header("📊 Current Demo Database")
    if st.checkbox("Show all demos"):
        st.dataframe(st.session_state.matcher.demos_df, use_container_width=True)

else:
    st.info("👆 Please upload your demo database or use the sample data to get started.")
    
    # Show expected format
    st.header("📋 Expected Spreadsheet Format")
    st.markdown("""
    Your spreadsheet should contain the following columns:
    
    - **Company Name**: Name of the company the demo was made for
    - **Industry**: Industry sector of the company
    - **Problem/Need**: Description of the customer's problem or need
    - **Solution Provided**: Description of the AI solution demonstrated
    - **Demo Type**: Type of demo (e.g., Live Demo, Video Demo, etc.)
    - **Demo Link/File**: Link or file path to the demo
    - **Success Rate**: Success rate or outcome of the demo
    - **Date**: Date when the demo was created
    
    *Note: Not all columns are required, but having more information will improve matching accuracy.*
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use descriptive language when describing customer needs for better matching results.")
