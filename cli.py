#!/usr/bin/env python3
"""
Command-line interface for the Demo Matcher
"""

import argparse
import sys
from demo_matcher import DemoMatcher

def main():
    parser = argparse.ArgumentParser(description="Find similar demos based on customer needs")
    parser.add_argument("customer_need", help="Description of customer's problem/need")
    parser.add_argument("-f", "--file", help="Path to demo database spreadsheet")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top matches to return")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = DemoMatcher()
    
    # Load data
    if args.sample:
        print("Creating and loading sample data...")
        sample_file = "sample_demo_database.csv"
        matcher.create_sample_data(sample_file)
        matcher.load_spreadsheet(sample_file)
    elif args.file:
        print(f"Loading data from {args.file}...")
        matcher.load_spreadsheet(args.file)
    else:
        print("Error: Please provide either a file path (-f) or use --sample")
        sys.exit(1)
    
    if matcher.demos_df is None:
        print("Error: No data loaded!")
        sys.exit(1)
    
    # Search for similar demos
    print(f"\nSearching for demos similar to: '{args.customer_need}'")
    print("-" * 60)
    
    analysis = matcher.get_detailed_analysis(args.customer_need, args.top_k)
    print(analysis)

if __name__ == "__main__":
    main()
