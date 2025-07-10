#!/usr/bin/env python3
"""
Test script to verify that exact text matching works correctly.
This will help diagnose why word-for-word matches aren't being found.
"""

import os
import sys
from demo_matcher import DemoMatcher
from openai_gpt_matcher import OpenAIGPTMatcher

def test_exact_matching():
    """Test both matchers with exact text from the CSV"""
    
    csv_file = "Copy of Master File Demos Database - Demos Database.csv"
    
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}")
        return
    
    print("=== TESTING EXACT MATCHING ===\n")
    
    # Test cases - exact text from the CSV
    test_cases = [
        "Navigate multiple sources of information, take hours to analyze data of products, losing time",
        "Losing time navigating information on client and product patterns across multiple platforms, resulting in reduced guest engagement and missed chances to personalize offers, increase sales, and retain customers",
        "Relying in hand-drawn drawings and manual processes to ensure sketches meet SOP compliance, prone to human error"
    ]
    
    # Test the sentence transformer matcher
    print("1. TESTING SENTENCE TRANSFORMER MATCHER")
    print("-" * 50)
    try:
        matcher = DemoMatcher(csv_file, ["Client Problem"])
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_case[:50]}...")
            
            # Debug the matching process
            debug_info = matcher.debug_matching(test_case)
            print(debug_info)
            
            # Get results
            results = matcher.find_similar_demos(test_case, top_k=3, min_score=0.0)
            print(f"\nFound {len(results)} results:")
            for result in results:
                score = result['similarity_score']
                company = result['demo_info'].get('Name/Client', 'Unknown')
                exact_match = result.get('debug_info', {}).get('exact_text_match', False)
                print(f"  - {company}: Score {score:.4f} (Exact: {exact_match})")
            
            print("\n" + "="*80 + "\n")
    
    except Exception as e:
        print(f"ERROR with SentenceTransformer matcher: {e}")
    
    # Test the OpenAI matcher (if API key available)
    print("2. TESTING OPENAI GPT MATCHER")
    print("-" * 50)
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("SKIPPED: No OPENAI_API_KEY found in environment")
        return
    
    try:
        openai_matcher = OpenAIGPTMatcher(csv_file, ["Client Problem"], openai_key)
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_case[:50]}...")
            
            # Debug the matching process
            debug_info = openai_matcher.debug_matching(test_case)
            print(debug_info)
            
            # Get results
            results = openai_matcher.find_best_demos(test_case, top_k=3)
            print(f"\nFound {len(results)} results:")
            for result in results:
                score = result.get('similarity_score', 0)
                company = result.get('demo_info', {}).get('Name/Client', 'Unknown')
                explanation = result.get('explanation', 'No explanation')
                print(f"  - {company}: Score {score:.4f}")
                print(f"    Explanation: {explanation}")
            
            print("\n" + "="*80 + "\n")
    
    except Exception as e:
        print(f"ERROR with OpenAI matcher: {e}")

if __name__ == "__main__":
    test_exact_matching()
