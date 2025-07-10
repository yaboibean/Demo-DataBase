"""
Debug Demo Matcher - Built to identify and fix matching issues
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DebugDemoMatcher:
    """
    Debug version of demo matcher to identify matching issues.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the debug demo matcher.
        
        Args:
            csv_path: Path to the CSV file containing demo data
        """
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.model = None
        self.embeddings = None
        self.problem_texts: List[str] = []
        
        # Initialize system
        self._setup()
    
    def _setup(self) -> None:
        """Setup the entire system."""
        try:
            logger.info("Starting debug demo matcher setup...")
            
            # Step 1: Load and clean data
            self._load_and_clean_data()
            
            # Step 2: Initialize model
            self._init_model()
            
            # Step 3: Extract problems and create embeddings
            self._extract_problems_and_embed()
            
            logger.info(f"Setup complete! Ready to match against {len(self.problem_texts)} demos.")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _load_and_clean_data(self) -> None:
        """Load CSV and perform basic cleaning."""
        logger.info(f"Loading data from: {self.csv_path}")
        
        try:
            # Load CSV
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} rows with columns: {list(self.df.columns)}")
            
            # Basic validation
            if len(self.df) == 0:
                raise ValueError("CSV file is empty")
            
            # Remove completely empty rows
            initial_count = len(self.df)
            self.df = self.df.dropna(how='all')
            final_count = len(self.df)
            
            logger.info(f"Data cleaned: {final_count} rows remaining (removed {initial_count - final_count} empty rows)")
            
        except Exception as e:
            logger.error(f"Failed to load/clean data: {e}")
            raise
    
    def _init_model(self) -> None:
        """Initialize the embedding model."""
        try:
            logger.info("Initializing embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _minimal_clean_text(self, text: Any) -> str:
        """
        Very minimal text cleaning to preserve original meaning.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Only remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _extract_problems_and_embed(self) -> None:
        """Extract problem texts and create embeddings with detailed logging."""
        logger.info("Extracting problem texts...")
        
        problems = []
        valid_indices = []
        debug_info = []
        
        for idx, row in self.df.iterrows():
            client_name = str(row.get('Name/Client', 'Unknown')).strip()
            
            # Get the Client Problem text
            raw_problem = row.get('Client Problem', '')
            clean_problem = self._minimal_clean_text(raw_problem)
            
            # Store debug info
            debug_entry = {
                'index': idx,
                'client': client_name,
                'raw_problem': raw_problem,
                'clean_problem': clean_problem,
                'has_problem': bool(clean_problem and len(clean_problem) > 5)
            }
            debug_info.append(debug_entry)
            
            # Only include if we have a meaningful problem description
            if clean_problem and len(clean_problem) > 5:
                problems.append(clean_problem)
                valid_indices.append(idx)
                
                # Log specific entries we care about
                if 'ARCO' in client_name or 'Pelican' in client_name:
                    logger.info(f"Found {client_name}: '{clean_problem}'")
        
        # Filter dataframe to only valid rows
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.problem_texts = problems
        
        logger.info(f"Extracted {len(self.problem_texts)} valid problem descriptions")
        
        # Show specific examples
        logger.info("Key problem texts:")
        for i, (problem, orig_idx) in enumerate(zip(self.problem_texts, valid_indices)):
            client = self.df.iloc[i].get('Name/Client', 'Unknown')
            if 'ARCO' in str(client) or 'Pelican' in str(client):
                logger.info(f"  [{i}] {client}: '{problem}'")
        
        if len(self.problem_texts) == 0:
            raise ValueError("No valid problem descriptions found in the data")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        if self.model is None:
            raise ValueError("Model not initialized")
            
        self.embeddings = self.model.encode(
            self.problem_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def debug_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Debug version of search with detailed logging.
        
        Args:
            query: User's search query/problem description
            top_k: Maximum number of results to return
            
        Returns:
            List of matching demos with debug info
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if self.model is None or self.embeddings is None:
            raise ValueError("System not properly initialized")
        
        # Clean the query minimally
        cleaned_query = self._minimal_clean_text(query)
        logger.info(f"Debug search for: '{cleaned_query}'")
        
        try:
            # Create query embedding
            query_embedding = self.model.encode(
                [cleaned_query],
                normalize_embeddings=True
            )
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Log similarity scores for key entries
            for i, (similarity, problem) in enumerate(zip(similarities, self.problem_texts)):
                client = self.df.iloc[i].get('Name/Client', 'Unknown')
                if 'ARCO' in str(client) or 'Pelican' in str(client):
                    logger.info(f"  Similarity to {client}: {similarity:.4f}")
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                similarity = similarities[idx]
                
                # Get demo data
                demo_row = self.df.iloc[idx]
                
                # Build result with debug info
                result = {
                    'rank': rank,
                    'similarity': float(similarity),
                    'index': idx,
                    'client_name': demo_row.get('Name/Client', 'Unknown'),
                    'industry': demo_row.get('Industry', 'Unknown'),
                    'problem': self.problem_texts[idx],
                    'solution': demo_row.get('Instalily AI Capabilities', ''),
                    'match_quality': self._get_quality_label(similarity),
                    'debug_info': {
                        'original_query': query,
                        'cleaned_query': cleaned_query,
                        'embedding_similarity': float(similarity),
                        'problem_length': len(self.problem_texts[idx]),
                        'client_name': demo_row.get('Name/Client', 'Unknown')
                    }
                }
                
                results.append(result)
            
            # Log top results
            logger.info(f"Top {min(5, len(results))} results:")
            for i, result in enumerate(results[:5]):
                logger.info(f"  {i+1}. {result['client_name']} ({result['similarity']:.4f}): '{result['problem'][:100]}...'")
            
            return results
            
        except Exception as e:
            logger.error(f"Debug search failed: {e}")
            raise
    
    def _get_quality_label(self, similarity: float) -> str:
        """Get quality label for similarity score."""
        if similarity >= 0.75:
            return "Excellent"
        elif similarity >= 0.6:
            return "Good"
        elif similarity >= 0.4:
            return "Fair"
        elif similarity >= 0.25:
            return "Weak"
        else:
            return "Poor"
    
    def test_specific_query(self, query: str) -> str:
        """Test a specific query and return detailed analysis."""
        results = self.debug_search(query, top_k=10)
        
        output = []
        output.append("=" * 80)
        output.append("DEBUG SEARCH RESULTS")
        output.append("=" * 80)
        output.append(f"Query: {query}")
        output.append(f"Found {len(results)} matches:\n")
        
        for i, result in enumerate(results):
            output.append(f"--- RANK {result['rank']} ---")
            output.append(f"Client: {result['client_name']}")
            output.append(f"Similarity: {result['similarity']:.6f}")
            output.append(f"Match Quality: {result['match_quality']}")
            output.append(f"Problem: {result['problem']}")
            output.append(f"Index: {result['index']}")
            output.append("")
            
            # Only show first 10 to avoid clutter
            if i >= 9:
                break
        
        return "\n".join(output)
    
    def find_exact_matches(self, query: str) -> List[Dict[str, Any]]:
        """Find entries that contain the exact query text."""
        exact_matches = []
        
        for i, problem in enumerate(self.problem_texts):
            if query.lower() in problem.lower():
                demo_row = self.df.iloc[i]
                exact_matches.append({
                    'index': i,
                    'client_name': demo_row.get('Name/Client', 'Unknown'),
                    'problem': problem,
                    'match_type': 'exact_substring'
                })
        
        return exact_matches
    
    def analyze_problem(self, query: str) -> str:
        """Analyze why a query might not be matching correctly."""
        analysis = []
        analysis.append("=" * 80)
        analysis.append("MATCHING PROBLEM ANALYSIS")
        analysis.append("=" * 80)
        analysis.append(f"Query: '{query}'")
        analysis.append("")
        
        # Find exact matches
        exact_matches = self.find_exact_matches(query)
        if exact_matches:
            analysis.append("EXACT SUBSTRING MATCHES FOUND:")
            for match in exact_matches:
                analysis.append(f"  • {match['client_name']}: '{match['problem']}'")
            analysis.append("")
        else:
            analysis.append("No exact substring matches found.")
            analysis.append("")
        
        # Get semantic matches
        semantic_matches = self.debug_search(query, top_k=5)
        analysis.append("TOP SEMANTIC MATCHES:")
        for match in semantic_matches:
            analysis.append(f"  • {match['client_name']} ({match['similarity']:.4f}): '{match['problem'][:100]}...'")
        
        analysis.append("")
        analysis.append("ANALYSIS:")
        if exact_matches and semantic_matches:
            exact_client = exact_matches[0]['client_name']
            semantic_client = semantic_matches[0]['client_name']
            
            if exact_client != semantic_client:
                analysis.append(f"❌ MISMATCH: Exact match is '{exact_client}' but top semantic match is '{semantic_client}'")
                analysis.append("This suggests an issue with the embedding or similarity calculation.")
            else:
                analysis.append(f"✅ MATCH: Both exact and semantic matching return '{exact_client}'")
        
        return "\n".join(analysis)


def main():
    """Main function for debugging."""
    try:
        print("Initializing Debug Demo Matcher...")
        
        # Initialize matcher
        csv_path = "Copy of Master File Demos Database - Demos Database.csv"
        matcher = DebugDemoMatcher(csv_path)
        
        # Test the specific problematic query
        problem_query = "The sales team struggles with disjointed workflows, missed follow-ups, and reliance on manual tools like Excel and email to manage high-value opportunities."
        
        print("\n" + "=" * 80)
        print("TESTING PROBLEMATIC QUERY")
        print("=" * 80)
        
        # Analyze the problem
        print(matcher.analyze_problem(problem_query))
        
        # Show detailed search results
        print("\n" + matcher.test_specific_query(problem_query))
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
