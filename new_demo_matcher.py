"""
Enhanced Demo Matcher - Completely rewritten for better performance and reliability
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDemoMatcher:
    """
    A robust demo matching system that finds similar demos based on customer problems/needs.
    """
    
    def __init__(self, csv_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Enhanced Demo Matcher.
        
        Args:
            csv_path: Path to the CSV file containing demo data
            embedding_model: Name of the SentenceTransformer model to use
        """
        self.csv_path = csv_path
        self.embedding_model_name = embedding_model
        self.demos_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        
        # Initialize the system
        self._load_data()
        self._initialize_model()
        self._create_embeddings()
        
        if self.demos_df is not None:
            logger.info(f"Demo matcher initialized with {len(self.demos_df)} demos")
    
    def _load_data(self) -> None:
        """Load and validate the CSV data."""
        try:
            self.demos_df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded CSV with {len(self.demos_df)} rows and {len(self.demos_df.columns)} columns")
            
            # Display available columns for debugging
            logger.info(f"Available columns: {list(self.demos_df.columns)}")
            
            # Check if we have the expected problem description column
            if 'Client Problem' not in self.demos_df.columns:
                raise ValueError("Required column 'Client Problem' not found in CSV")
            
            # Remove rows where Client Problem is empty or NaN
            initial_count = len(self.demos_df)
            self.demos_df = self.demos_df.dropna(subset=['Client Problem'])
            self.demos_df = self.demos_df[self.demos_df['Client Problem'].str.strip() != '']
            final_count = len(self.demos_df)
            
            if final_count == 0:
                raise ValueError("No valid demo data found with non-empty 'Client Problem' field")
            
            logger.info(f"Filtered to {final_count} demos with valid problems (removed {initial_count - final_count})")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _clean_and_normalize_text(self, text: str) -> str:
        """
        Clean and normalize text for better embedding quality.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common noise patterns
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Ensure the text isn't empty after cleaning
        text = text.strip()
        
        return text
    
    def _create_embeddings(self) -> None:
        """Create embeddings for all demo problems."""
        if self.demos_df is None or self.model is None:
            raise ValueError("Data or model not initialized")
            
        try:
            # Extract and clean all client problems
            problem_texts = []
            valid_indices = []
            
            for idx, row in self.demos_df.iterrows():
                problem_text = self._clean_and_normalize_text(row['Client Problem'])
                if problem_text:  # Only include non-empty problems
                    problem_texts.append(problem_text)
                    valid_indices.append(idx)
            
            if not problem_texts:
                raise ValueError("No valid problem texts found for embedding")
            
            # Filter dataframe to only include rows with valid problems
            self.demos_df = self.demos_df.loc[valid_indices].reset_index(drop=True)
            
            logger.info(f"Creating embeddings for {len(problem_texts)} problem descriptions...")
            
            # Create embeddings
            self.embeddings = self.model.encode(
                problem_texts, 
                show_progress_bar=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    def find_similar_demos(
        self, 
        customer_need: str, 
        top_k: int = 5, 
        min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Find demos most similar to the customer's need.
        
        Args:
            customer_need: Description of the customer's problem/need
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity score to include in results
            
        Returns:
            List of matching demos with similarity scores and metadata
        """
        if not customer_need or not isinstance(customer_need, str):
            raise ValueError("customer_need must be a non-empty string")
        
        # Clean the customer need
        cleaned_need = self._clean_and_normalize_text(customer_need)
        if not cleaned_need:
            logger.warning("Customer need is empty after cleaning")
            return []
        
        try:
            # Create embedding for customer need
            need_embedding = self.model.encode(
                [cleaned_need], 
                normalize_embeddings=True
            )
            
            # Calculate similarities
            similarities = cosine_similarity(need_embedding, self.embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1]
            
            results = []
            for rank, idx in enumerate(top_indices[:top_k], 1):
                similarity_score = similarities[idx]
                
                # Skip if below minimum threshold
                if similarity_score < min_similarity:
                    continue
                
                # Get demo information
                demo_row = self.demos_df.iloc[idx]
                
                result = {
                    'rank': rank,
                    'similarity_score': float(similarity_score),
                    'demo_info': {
                        'name': demo_row.get('Name/Client', 'Unknown'),
                        'industry': demo_row.get('Industry', 'Unknown'),
                        'client_problem': demo_row.get('Client Problem', ''),
                        'solution': demo_row.get('Instalily AI Capabilities', demo_row.get('Solution', '')),
                        'benefits': demo_row.get('Benefit to Client', ''),
                        'demo_link': demo_row.get('Demo link ', demo_row.get('Portal Demo Link', '')),
                        'date': demo_row.get('Date Uploaded', ''),
                    },
                    'match_quality': self._assess_match_quality(similarity_score),
                    'debug_info': {
                        'original_need': customer_need,
                        'cleaned_need': cleaned_need,
                        'original_problem': demo_row.get('Client Problem', ''),
                        'cleaned_problem': self._clean_and_normalize_text(demo_row.get('Client Problem', ''))
                    }
                }
                
                results.append(result)
            
            logger.info(f"Found {len(results)} similar demos for query: '{customer_need[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar demos: {e}")
            raise
    
    def _assess_match_quality(self, similarity_score: float) -> str:
        """
        Assess the quality of a match based on similarity score.
        
        Args:
            similarity_score: Cosine similarity score
            
        Returns:
            Quality assessment string
        """
        if similarity_score >= 0.8:
            return "Excellent Match"
        elif similarity_score >= 0.6:
            return "Good Match"
        elif similarity_score >= 0.4:
            return "Fair Match"
        elif similarity_score >= 0.2:
            return "Weak Match"
        else:
            return "Poor Match"
    
    def get_formatted_results(
        self, 
        customer_need: str, 
        top_k: int = 3,
        include_debug: bool = False
    ) -> str:
        """
        Get formatted results for display.
        
        Args:
            customer_need: Customer's problem description
            top_k: Number of results to return
            include_debug: Whether to include debug information
            
        Returns:
            Formatted string with results
        """
        try:
            results = self.find_similar_demos(customer_need, top_k)
            
            if not results:
                return f"No similar demos found for: '{customer_need}'\n\nTry rephrasing your query or reducing the minimum similarity threshold."
            
            output = []
            output.append("=" * 80)
            output.append("DEMO MATCHING RESULTS")
            output.append("=" * 80)
            output.append(f"Customer Need: {customer_need}")
            output.append(f"Found {len(results)} matching demos:\n")
            
            for result in results:
                demo = result['demo_info']
                output.append(f"--- RANK {result['rank']} ({result['match_quality']}) ---")
                output.append(f"Similarity Score: {result['similarity_score']:.3f}")
                output.append(f"Client: {demo['name']}")
                output.append(f"Industry: {demo['industry']}")
                output.append(f"Problem: {demo['client_problem']}")
                
                if demo['solution']:
                    output.append(f"Solution: {demo['solution']}")
                
                if demo['benefits']:
                    output.append(f"Benefits: {demo['benefits']}")
                
                if demo['demo_link']:
                    output.append(f"Demo Link: {demo['demo_link']}")
                
                if demo['date']:
                    output.append(f"Date: {demo['date']}")
                
                if include_debug:
                    debug = result['debug_info']
                    output.append("\nDEBUG INFO:")
                    output.append(f"  Original Need: '{debug['original_need']}'")
                    output.append(f"  Cleaned Need: '{debug['cleaned_need']}'")
                    output.append(f"  Original Problem: '{debug['original_problem']}'")
                    output.append(f"  Cleaned Problem: '{debug['cleaned_problem']}'")
                
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return f"Error generating results: {e}"
    
    def get_data_summary(self) -> str:
        """
        Get a summary of the loaded data for debugging.
        
        Returns:
            Summary string
        """
        if self.demos_df is None:
            return "No data loaded"
        
        summary = []
        summary.append("=" * 60)
        summary.append("DATA SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total demos: {len(self.demos_df)}")
        summary.append(f"Columns: {list(self.demos_df.columns)}")
        summary.append("")
        
        # Show sample problems
        summary.append("SAMPLE CLIENT PROBLEMS:")
        for idx, row in self.demos_df.head(5).iterrows():
            client = row.get('Name/Client', 'Unknown')
            problem = row.get('Client Problem', '')[:100] + "..." if len(str(row.get('Client Problem', ''))) > 100 else row.get('Client Problem', '')
            summary.append(f"  {client}: {problem}")
        
        summary.append("")
        
        # Show industries
        if 'Industry' in self.demos_df.columns:
            industries = self.demos_df['Industry'].value_counts().head(10)
            summary.append("TOP INDUSTRIES:")
            for industry, count in industries.items():
                summary.append(f"  {industry}: {count}")
        
        return "\n".join(summary)
    
    def test_matching(self, test_queries: List[str] = None) -> str:
        """
        Test the matching system with sample queries.
        
        Args:
            test_queries: List of test queries, uses defaults if None
            
        Returns:
            Test results string
        """
        if test_queries is None:
            test_queries = [
                "Customer support automation",
                "Inventory management",
                "Document processing",
                "Predictive maintenance",
                "Sales optimization"
            ]
        
        results = []
        results.append("=" * 60)
        results.append("MATCHING SYSTEM TEST")
        results.append("=" * 60)
        
        for query in test_queries:
            results.append(f"\nTesting: '{query}'")
            try:
                matches = self.find_similar_demos(query, top_k=2)
                if matches:
                    for match in matches:
                        results.append(f"  → {match['demo_info']['name']} (Score: {match['similarity_score']:.3f})")
                else:
                    results.append("  → No matches found")
            except Exception as e:
                results.append(f"  → Error: {e}")
        
        return "\n".join(results)


def main():
    """Main function for testing the enhanced demo matcher."""
    try:
        # Initialize the matcher
        csv_path = "Copy of Master File Demos Database - Demos Database.csv"
        matcher = EnhancedDemoMatcher(csv_path)
        
        # Print data summary
        print(matcher.get_data_summary())
        print("\n")
        
        # Test the system
        print(matcher.test_matching())
        print("\n")
        
        # Interactive demo
        print("=" * 80)
        print("INTERACTIVE DEMO MATCHING")
        print("=" * 80)
        
        sample_need = "We need help with customer service automation and handling support tickets efficiently"
        print(matcher.get_formatted_results(sample_need, top_k=3))
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
