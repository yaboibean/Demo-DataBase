"""
Demo Matcher V2 - Rebuilt from scratch with simplified, robust architecture
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


class DemoMatcherV2:
    """
    Simplified and robust demo matching system.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the demo matcher.
        
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
            logger.info("Starting demo matcher setup...")
            
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
            
            # Check for required columns (at least one should exist)
            required_cols = ['Client Problem', 'Name/Client']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if len(missing_cols) == len(required_cols):
                raise ValueError(f"None of the required columns found: {required_cols}")
            
            # Remove completely empty rows
            initial_count = len(self.df)
            self.df = self.df.dropna(how='all')
            final_count = len(self.df)
            
            if final_count == 0:
                raise ValueError("No data remaining after removing empty rows")
            
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
    
    def _clean_text(self, text: Any) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text).strip()
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'[^\w\s.,!?()-]', ' ', text)  # Remove special chars except basic punctuation
        text = text.strip()
        
        return text
    
    def _extract_problems_and_embed(self) -> None:
        """Extract problem texts and create embeddings."""
        logger.info("Extracting problem texts...")
        
        # Extract problem texts with fallbacks
        problems = []
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            # Try different columns for problem description
            problem_text = ""
            
            # Primary: Client Problem column
            if 'Client Problem' in self.df.columns and pd.notna(row.get('Client Problem')):
                problem_text = self._clean_text(row['Client Problem'])
            
            # Fallback 1: Use Case Explained
            if not problem_text and 'Use Case Explained' in self.df.columns:
                problem_text = self._clean_text(row.get('Use Case Explained'))
            
            # Fallback 2: Solution column
            if not problem_text and 'Solution' in self.df.columns:
                problem_text = self._clean_text(row.get('Solution'))
            
            # Fallback 3: Instalily AI Capabilities
            if not problem_text and 'Instalily AI Capabilities' in self.df.columns:
                problem_text = self._clean_text(row.get('Instalily AI Capabilities'))
            
            # Fallback 4: Benefit to Client
            if not problem_text and 'Benefit to Client' in self.df.columns:
                problem_text = self._clean_text(row.get('Benefit to Client'))
            
            # Fallback 5: Use client name + industry as context
            if not problem_text:
                client = self._clean_text(row.get('Name/Client', ''))
                industry = self._clean_text(row.get('Industry', ''))
                if client or industry:
                    problem_text = f"{client} {industry}".strip()
            
            # Only include if we have some text
            if problem_text and len(problem_text) > 3:  # At least 4 characters
                problems.append(problem_text)
                valid_indices.append(idx)
        
        # Filter dataframe to only valid rows
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.problem_texts = problems
        
        logger.info(f"Extracted {len(self.problem_texts)} valid problem descriptions")
        
        if len(self.problem_texts) == 0:
            raise ValueError("No valid problem descriptions found in the data")
        
        # Show some examples
        logger.info("Sample problem texts:")
        for i, text in enumerate(self.problem_texts[:3]):
            logger.info(f"  {i+1}: {text[:100]}...")
        
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
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar demos.
        
        Args:
            query: User's search query/problem description
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching demos with metadata
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if self.model is None or self.embeddings is None:
            raise ValueError("System not properly initialized")
        
        # Clean the query
        cleaned_query = self._clean_text(query)
        if not cleaned_query:
            logger.warning("Query is empty after cleaning")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.model.encode(
                [cleaned_query],
                normalize_embeddings=True
            )
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                similarity = similarities[idx]
                
                # Skip if below threshold
                if similarity < min_similarity:
                    continue
                
                # Get demo data
                demo_row = self.df.iloc[idx]
                
                # Build result
                result = {
                    'rank': rank,
                    'similarity': float(similarity),
                    'similarity_score': float(similarity),  # Add for compatibility
                    'client_name': demo_row.get('Name/Client', 'Unknown'),
                    'industry': demo_row.get('Industry', 'Unknown'),
                    'problem': self.problem_texts[idx],
                    'client_problem': self.problem_texts[idx],  # Add for compatibility
                    'solution': self._clean_text(demo_row.get('Instalily AI Capabilities', '')),
                    'benefits': self._clean_text(demo_row.get('Benefit to Client', '')),
                    'demo_link': demo_row.get('Demo link ', ''),  # Use the rightmost 'Demo link ' column
                    'date': demo_row.get('Date Uploaded', ''),
                    'match_quality': self._get_quality_label(similarity),
                    'explanation': f"{self._get_quality_label(similarity)} match with {similarity:.3f} similarity",
                    'demo_info': demo_row.to_dict()  # Add full demo info for compatibility
                }
                
                results.append(result)
            
            logger.info(f"Found {len(results)} matches for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
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
    
    def format_results(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted search results.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Formatted results string
        """
        try:
            results = self.search(query, top_k)
            
            if not results:
                return f"No demos found for: '{query}'\n\nTry a different search term."
            
            output = []
            output.append("=" * 80)
            output.append("DEMO SEARCH RESULTS")
            output.append("=" * 80)
            output.append(f"Query: {query}")
            output.append(f"Found {len(results)} matching demos:\n")
            
            for result in results:
                output.append(f"--- RANK {result['rank']} ({result['match_quality']} Match) ---")
                output.append(f"Similarity: {result['similarity']:.3f}")
                output.append(f"Client: {result['client_name']}")
                output.append(f"Industry: {result['industry']}")
                output.append(f"Problem: {result['problem']}")
                
                if result['solution']:
                    output.append(f"Solution: {result['solution']}")
                
                if result['benefits']:
                    output.append(f"Benefits: {result['benefits']}")
                
                if result['demo_link']:
                    output.append(f"Demo Link: {result['demo_link']}")
                
                if result['date']:
                    output.append(f"Date: {result['date']}")
                
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return f"Error: {e}"
    
    def get_stats(self) -> str:
        """Get system statistics."""
        if self.df is None:
            return "System not initialized"
        
        stats = []
        stats.append("=" * 60)
        stats.append("DEMO MATCHER STATISTICS")
        stats.append("=" * 60)
        stats.append(f"Total demos loaded: {len(self.df)}")
        stats.append(f"Valid problem descriptions: {len(self.problem_texts)}")
        stats.append(f"Available columns: {list(self.df.columns)}")
        stats.append("")
        
        # Industry breakdown
        if 'Industry' in self.df.columns:
            industry_counts = self.df['Industry'].value_counts().head(10)
            stats.append("Top Industries:")
            for industry, count in industry_counts.items():
                stats.append(f"  • {industry}: {count}")
            stats.append("")
        
        # Sample problems
        stats.append("Sample Problems:")
        for i, problem in enumerate(self.problem_texts[:5], 1):
            client = self.df.iloc[i-1].get('Name/Client', 'Unknown')
            short_problem = problem[:80] + "..." if len(problem) > 80 else problem
            stats.append(f"  {i}. {client}: {short_problem}")
        
        return "\n".join(stats)
    
    def test_search(self) -> str:
        """Test the search functionality."""
        test_queries = [
            "customer service automation",
            "inventory management system",
            "document processing",
            "supply chain optimization",
            "contract management",
            "data analysis and insights"
        ]
        
        results = []
        results.append("=" * 60)
        results.append("SEARCH FUNCTIONALITY TEST")
        results.append("=" * 60)
        
        for query in test_queries:
            results.append(f"\nTesting: '{query}'")
            try:
                matches = self.search(query, top_k=2)
                if matches:
                    for match in matches:
                        results.append(f"  ✓ {match['client_name']} ({match['similarity']:.3f})")
                else:
                    results.append("  ✗ No matches found")
            except Exception as e:
                results.append(f"  ✗ Error: {e}")
        
        return "\n".join(results)


def main():
    """Main function for testing."""
    try:
        print("Initializing Demo Matcher V2...")
        
        # Initialize matcher
        csv_path = "Copy of Master File Demos Database - Demos Database.csv"
        matcher = DemoMatcherV2(csv_path)
        
        # Show statistics
        print(matcher.get_stats())
        print()
        
        # Test functionality
        print(matcher.test_search())
        print()
        
        # Interactive example
        print("=" * 80)
        print("EXAMPLE SEARCH")
        print("=" * 80)
        
        example_query = "help with customer support and ticket management"
        print(matcher.format_results(example_query, top_k=3))
        
        # Allow interactive usage
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("Enter your search queries (type 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\nSearch query: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    print(matcher.format_results(user_input, top_k=3))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
