"""
Ultra-Robust Demo Matcher - Final version with enhanced accuracy and debugging
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltraRobustDemoMatcher:
    """
    Ultra-robust demo matcher with enhanced accuracy and debugging capabilities.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the ultra-robust demo matcher.
        
        Args:
            csv_path: Path to the CSV file containing demo data
        """
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.valid_problems: List[str] = []
        self.valid_indices: List[int] = []
        
        # Setup the system
        self._full_setup()
    
    def _full_setup(self) -> None:
        """Complete system setup with error handling."""
        try:
            logger.info("🚀 Starting Ultra-Robust Demo Matcher setup...")
            
            # Load data
            self._load_data()
            
            # Initialize model
            self._setup_model()
            
            # Process problems and create embeddings
            self._process_and_embed()
            
            logger.info(f"✅ Setup complete! Loaded {len(self.valid_problems)} valid demos.")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self) -> None:
        """Load and validate CSV data."""
        logger.info(f"📂 Loading data from: {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            
            if self.df.empty:
                raise ValueError("CSV file is empty")
            
            logger.info(f"📊 Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            logger.info(f"📋 Columns: {list(self.df.columns)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _setup_model(self) -> None:
        """Initialize the embedding model."""
        logger.info("🧠 Initializing embedding model...")
        
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def _clean_text(self, text: Any) -> str:
        """
        Clean text while preserving meaning.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and basic cleaning
        text = str(text).strip()
        
        # Normalize whitespace
        import re
        text = re.sub(r'\\s+', ' ', text)
        
        return text
    
    def _process_and_embed(self) -> None:
        """Process problems and create embeddings."""
        logger.info("🔍 Processing problem descriptions...")
        
        if self.df is None:
            raise ValueError("DataFrame not initialized")
        
        problems = []
        indices = []
        
        # Process each row
        for idx, row in self.df.iterrows():
            client_name = self._clean_text(row.get('Name/Client', ''))
            
            # Get problem text from Client Problem column
            problem_text = self._clean_text(row.get('Client Problem', ''))
            
            # Only include if we have meaningful problem text
            if problem_text and len(problem_text) > 5:
                problems.append(problem_text)
                indices.append(idx)
                
                # Log important entries
                if any(keyword in client_name.upper() for keyword in ['ARCO', 'PELICAN']):
                    logger.info(f"📝 Found {client_name}: '{problem_text}'")
        
        # Store results
        self.valid_problems = problems
        self.valid_indices = indices
        
        # Filter dataframe
        self.df = self.df.iloc[indices].reset_index(drop=True)
        
        logger.info(f"✅ Processed {len(self.valid_problems)} valid problem descriptions")
        
        if not self.valid_problems:
            raise ValueError("No valid problem descriptions found")
        
        # Create embeddings
        logger.info("🔢 Creating embeddings...")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.embeddings = self.model.encode(
            self.valid_problems,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        logger.info(f"✅ Created embeddings with shape: {self.embeddings.shape}")
        
        # Verify key entries
        self._verify_key_entries()
    
    def _verify_key_entries(self) -> None:
        """Verify that key entries (ARCO, Pelican) are correctly loaded."""
        logger.info("🔍 Verifying key entries...")
        
        if self.df is None:
            logger.warning("⚠️ DataFrame not initialized")
            return
        
        arco_found = False
        pelican_found = False
        
        for i, row in self.df.iterrows():
            client_name = str(row.get('Name/Client', ''))
            
            if 'ARCO' in client_name.upper():
                arco_found = True
                logger.info(f"✅ ARCO found at index {i}: '{self.valid_problems[int(i)]}'")
            
            if 'PELICAN' in client_name.upper():
                pelican_found = True
                logger.info(f"✅ Pelican found at index {i}: '{self.valid_problems[int(i)]}'")
        
        if not arco_found:
            logger.warning("⚠️ ARCO entry not found in processed data")
        if not pelican_found:
            logger.warning("⚠️ Pelican entry not found in processed data")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar demos.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching demos
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Clean query
        clean_query = self._clean_text(query)
        logger.info(f"🔍 Searching for: '{clean_query}'")
        
        try:
            # Create query embedding
            if self.model is None:
                raise ValueError("Model not initialized")
            
            query_embedding = self.model.encode(
                [clean_query],
                normalize_embeddings=True
            )
            
            # Calculate similarities
            if self.embeddings is None:
                raise ValueError("Embeddings not created")
            
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Log similarities for key entries
            if self.df is not None:
                for i, similarity in enumerate(similarities):
                    client_name = str(self.df.iloc[i].get('Name/Client', ''))
                    if any(keyword in client_name.upper() for keyword in ['ARCO', 'PELICAN']):
                        logger.info(f"📊 Similarity to {client_name}: {similarity:.6f}")
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                similarity = similarities[idx]
                
                if self.df is not None:
                    row = self.df.iloc[idx]
                    
                    result = {
                        'rank': rank,
                        'similarity': float(similarity),
                        'client_name': row.get('Name/Client', 'Unknown'),
                        'industry': row.get('Industry', 'Unknown'),
                        'problem': self.valid_problems[idx],
                        'solution': self._clean_text(row.get('Instalily AI Capabilities', '')),
                        'benefits': self._clean_text(row.get('Benefit to Client', '')),
                        'demo_link': row.get('Portal Demo Link', ''),
                        'date': row.get('Date Uploaded', ''),
                        'quality': self._get_quality_label(similarity)
                    }
                    
                    results.append(result)
            
            # Log results
            logger.info(f"📋 Top {min(3, len(results))} results:")
            for i, result in enumerate(results[:3]):
                logger.info(f"  {i+1}. {result['client_name']} ({result['similarity']:.6f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _get_quality_label(self, similarity: float) -> str:
        """Get quality label for similarity score."""
        if similarity >= 0.8:
            return "Excellent"
        elif similarity >= 0.6:
            return "Good"
        elif similarity >= 0.4:
            return "Fair"
        elif similarity >= 0.2:
            return "Weak"
        else:
            return "Poor"
    
    def format_results(self, query: str, top_k: int = 3) -> str:
        """
        Format search results for display.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Formatted results string
        """
        try:
            results = self.search(query, top_k)
            
            if not results:
                return f"No results found for: '{query}'"
            
            output = []
            output.append("=" * 80)
            output.append("🎯 DEMO SEARCH RESULTS")
            output.append("=" * 80)
            output.append(f"Query: {query}")
            output.append(f"Found {len(results)} matching demos:\\n")
            
            for result in results:
                output.append(f"--- 🏆 RANK {result['rank']} ({result['quality']} Match) ---")
                output.append(f"📊 Similarity: {result['similarity']:.6f}")
                output.append(f"🏢 Client: {result['client_name']}")
                output.append(f"🏭 Industry: {result['industry']}")
                output.append(f"❓ Problem: {result['problem']}")
                
                if result['solution']:
                    output.append(f"💡 Solution: {result['solution']}")
                
                if result['benefits']:
                    output.append(f"💰 Benefits: {result['benefits']}")
                
                if result['demo_link']:
                    output.append(f"🔗 Demo Link: {result['demo_link']}")
                
                if result['date']:
                    output.append(f"📅 Date: {result['date']}")
                
                output.append("")
            
            return "\\n".join(output)
            
        except Exception as e:
            logger.error(f"❌ Format results failed: {e}")
            return f"Error: {e}"
    
    def test_problematic_query(self) -> str:
        """Test the specific problematic query."""
        problematic_query = "The sales team struggles with disjointed workflows, missed follow-ups, and reliance on manual tools like Excel and email to manage high-value opportunities."
        
        logger.info("🧪 Testing problematic query...")
        
        return self.format_results(problematic_query, top_k=5)
    
    def get_stats(self) -> str:
        """Get system statistics."""
        stats = []
        stats.append("=" * 60)
        stats.append("📊 SYSTEM STATISTICS")
        stats.append("=" * 60)
        stats.append(f"Total demos: {len(self.df) if self.df is not None else 0}")
        stats.append(f"Valid problems: {len(self.valid_problems)}")
        stats.append(f"Embedding dimensions: {self.embeddings.shape[1] if self.embeddings is not None else 0}")
        stats.append("")
        
        if self.df is not None:
            # Show sample entries
            stats.append("Sample entries:")
            for i in range(min(5, len(self.df))):
                client = self.df.iloc[i].get('Name/Client', 'Unknown')
                problem = self.valid_problems[i][:80] + "..." if len(self.valid_problems[i]) > 80 else self.valid_problems[i]
                stats.append(f"  {i+1}. {client}: {problem}")
        
        return "\\n".join(stats)


def main():
    """Main function for testing."""
    try:
        print("🚀 Initializing Ultra-Robust Demo Matcher...")
        
        # Initialize
        matcher = UltraRobustDemoMatcher("Copy of Master File Demos Database - Demos Database.csv")
        
        # Show stats
        print(matcher.get_stats())
        print()
        
        # Test problematic query
        print("🧪 Testing the problematic query...")
        print(matcher.test_problematic_query())
        
        # Test additional queries
        print("\\n" + "=" * 80)
        print("🎯 ADDITIONAL TESTS")
        print("=" * 80)
        
        test_queries = [
            "customer service automation",
            "inventory management",
            "supply chain optimization"
        ]
        
        for query in test_queries:
            print(f"\\nTesting: '{query}'")
            try:
                results = matcher.search(query, top_k=2)
                for result in results:
                    print(f"  → {result['client_name']} ({result['similarity']:.4f})")
            except Exception as e:
                print(f"  → Error: {e}")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
