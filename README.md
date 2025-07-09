# AI Demo Database Matcher 🎯

A powerful tool to find the most similar past demos based on current customer needs using semantic similarity matching.

## Features

- **Semantic Matching**: Uses AI embeddings to find contextually similar demos
- **Multiple Interfaces**: Web UI, command-line, and Python API
- **Flexible Data Format**: Works with CSV and Excel files
- **Detailed Analysis**: Provides similarity scores and detailed comparisons
- **Export Results**: Download matching results as CSV

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Interface

```bash
streamlit run streamlit_app.py
```

### 3. Use Sample Data (for testing)

The system includes sample demo data. Click "Use Sample Database" in the web interface or use the `--sample` flag in CLI.

## Usage

### Web Interface

1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Upload your demo database spreadsheet or use sample data
3. Enter the customer's problem/need description
4. Get ranked similar demos with similarity scores

### Command Line Interface

```bash
# Using your own spreadsheet
python cli.py "Customer needs AI for inventory management" -f your_demo_database.csv

# Using sample data
python cli.py "Customer needs AI for inventory management" --sample

# Get top 3 matches
python cli.py "Customer needs AI for inventory management" --sample -k 3
```

### Python API

```python
from demo_matcher import DemoMatcher

# Initialize matcher
matcher = DemoMatcher("your_demo_database.csv")

# Find similar demos
customer_need = "We need AI for customer support automation"
similar_demos = matcher.find_similar_demos(customer_need, top_k=5)

# Get detailed analysis
analysis = matcher.get_detailed_analysis(customer_need)
print(analysis)
```

## Spreadsheet Format

Your demo database should include these columns (not all required):

- **Company Name**: Name of the company
- **Industry**: Industry sector
- **Problem/Need**: Customer's problem description
- **Solution Provided**: AI solution demonstrated
- **Demo Type**: Type of demo (Live Demo, Video, etc.)
- **Demo Link/File**: Link or path to demo
- **Success Rate**: Demo success rate
- **Date**: Demo creation date

## How It Works

1. **Text Processing**: Combines problem description, solution, and context
2. **Semantic Embeddings**: Uses sentence-transformers to create meaning-based vectors
3. **Similarity Matching**: Computes cosine similarity between customer need and past demos
4. **Ranking**: Returns top matches with similarity scores

## Example Output

```
=== DEMO MATCHING ANALYSIS ===
Customer Need: We need an AI system to help with customer support

--- RANK 1 (Similarity: 0.845) ---
Company: RetailMax
Industry: Retail
Problem: Customer service chatbot for e-commerce platform
Solution: Conversational AI that handles customer inquiries and support tickets
Demo Type: Chatbot Interaction Demo
Success Rate: 78%
```

## Tips for Better Matching

- Use detailed descriptions of customer needs
- Include industry context when possible
- Mention specific use cases or requirements
- Keep demo database updated with comprehensive information

## Files

- `demo_matcher.py`: Core matching logic
- `streamlit_app.py`: Web interface
- `cli.py`: Command-line interface
- `requirements.txt`: Python dependencies
- `sample_demo_database.csv`: Generated sample data

## License

This tool is designed for internal use at your AI agent software company.
