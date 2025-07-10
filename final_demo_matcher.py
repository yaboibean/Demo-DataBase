"""
Final Demo Matcher - Streamlined and accurate version
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class FinalDemoMatcher:
    """
    Final, streamlined demo matcher that focuses on accuracy.
    """
    
    def __init__(self, csv_path: str):
        """Initialize the matcher."""
        self.csv_path = csv_path
        self.setup()
    
    def setup(self):
        """Setup the system."""
        print("🚀 Initializing Final Demo Matcher...")
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        print(f"📊 Loaded {len(self.df)} rows")
        
        # Initialize model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("🧠 Model initialized")
        
        # Process problems
        self.problems = []
        self.valid_rows = []
        
        for idx, row in self.df.iterrows():
            problem = str(row.get('Client Problem', '')).strip()
            if problem and len(problem) > 5:
                self.problems.append(problem)
                self.valid_rows.append(idx)
        
        # Filter dataframe
        self.df = self.df.iloc[self.valid_rows].reset_index(drop=True)
        
        print(f"✅ Processing {len(self.problems)} valid problems")
        
        # Create embeddings
        self.embeddings = self.model.encode(
            self.problems,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        print(f"🔢 Created embeddings: {self.embeddings.shape}")
        print("✅ Setup complete!")
    
    def search(self, query: str, top_k: int = 3):
        """Search for similar demos."""
        print(f"\\n🔍 Searching for: '{query}'")
        
        # Create query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            row = self.df.iloc[idx]
            
            result = {
                'rank': rank,
                'similarity': similarity,
                'similarity_score': similarity,  # Add for compatibility
                'client': row.get('Name/Client', 'Unknown'),
                'client_name': row.get('Name/Client', 'Unknown'),  # Add for compatibility
                'industry': row.get('Industry', 'Unknown'),
                'problem': self.problems[idx],
                'client_problem': self.problems[idx],  # Add for compatibility
                'solution': row.get('Instalily AI Capabilities', ''),
                'benefits': row.get('Benefit to Client', ''),
                'demo_link': row.get('Demo link ', ''),  # Use the rightmost 'Demo link ' column
                'date': row.get('Date Uploaded', ''),
                'explanation': f"Similarity: {similarity:.3f} - {self.problems[idx][:100]}{'...' if len(self.problems[idx]) > 100 else ''}",
                'demo_info': row.to_dict()  # Add full demo info for compatibility
            }
            results.append(result)
        
        return results
    
    def display_results(self, query: str, top_k: int = 3):
        """Display formatted results."""
        results = self.search(query, top_k)
        
        print("\\n" + "="*80)
        print("🎯 DEMO SEARCH RESULTS")
        print("="*80)
        print(f"Query: {query}")
        print(f"Found {len(results)} matches:\\n")
        
        for result in results:
            print(f"--- 🏆 RANK {result['rank']} ---")
            print(f"📊 Similarity: {result['similarity']:.6f}")
            print(f"🏢 Client: {result['client']}")
            print(f"🏭 Industry: {result['industry']}")
            print(f"❓ Problem: {result['problem']}")
            
            if result['solution']:
                print(f"💡 Solution: {result['solution']}")
            
            if result['benefits']:
                print(f"💰 Benefits: {result['benefits']}")
            
            if result['demo_link']:
                print(f"🔗 Demo Link: {result['demo_link']}")
            
            if result['date']:
                print(f"📅 Date: {result['date']}")
            
            print()
        
        return results
    
    def test_problematic_query(self):
        """Test the specific problematic query."""
        query = "The sales team struggles with disjointed workflows, missed follow-ups, and reliance on manual tools like Excel and email to manage high-value opportunities."
        
        print("\\n🧪 TESTING PROBLEMATIC QUERY")
        print("="*80)
        
        results = self.display_results(query, top_k=5)
        
        # Verify ARCO is #1
        if results and 'ARCO' in results[0]['client']:
            print("✅ SUCCESS: ARCO Demo is ranked #1!")
        else:
            print("❌ ISSUE: ARCO Demo is not ranked #1")
        
        return results


def main():
    """Main function."""
    try:
        # Initialize matcher
        matcher = FinalDemoMatcher("Copy of Master File Demos Database - Demos Database.csv")
        
        # Test the problematic query
        matcher.test_problematic_query()
        
        # Interactive mode
        print("\\n" + "="*80)
        print("🎮 INTERACTIVE MODE")
        print("="*80)
        print("Enter search queries (type 'quit' to exit):")
        
        while True:
            try:
                query = input("\\n🔍 Search: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query:
                    matcher.display_results(query, top_k=3)
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
