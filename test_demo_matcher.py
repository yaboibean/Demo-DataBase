"""
Demo Matcher V2 - Simple Test Script
"""

from demo_matcher_v2 import DemoMatcherV2

def main():
    """Simple test function."""
    print("Initializing Demo Matcher V2...")
    
    # Initialize matcher
    csv_path = "/Users/arjansingh/Downloads/Demo Database/Copy of Master File Demos Database - Demos Database.csv"
    matcher = DemoMatcherV2(csv_path)
    
    # Test queries
    test_queries = [
        "customer service automation and support tickets",
        "inventory management and warehouse optimization", 
        "sales process automation and CRM integration",
        "document processing and workflow automation",
        "supply chain management and logistics",
        "contract management and legal compliance"
    ]
    
    print("\n" + "="*80)
    print("TESTING DEMO MATCHER V2")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        print("-" * 50)
        
        try:
            results = matcher.search(query, top_k=2)
            
            if results:
                for result in results:
                    print(f"   • {result['client_name']} ({result['match_quality']} - {result['similarity']:.3f})")
                    print(f"     Industry: {result['industry']}")
                    print(f"     Problem: {result['problem'][:80]}...")
                    if result['demo_link']:
                        print(f"     Demo: {result['demo_link']}")
                    print()
            else:
                print("   No matches found")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "="*80)
    print("DETAILED EXAMPLE")
    print("="*80)
    
    example_query = "Sales reps spend too much time on manual data entry and can't focus on building relationships with customers"
    print(f"\nQuery: {example_query}")
    print(matcher.format_results(example_query, top_k=3))

if __name__ == "__main__":
    main()
