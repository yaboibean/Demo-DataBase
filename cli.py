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
    parser.add_argument("-k", "--top_k", type=int, default=2, help="Number of top matches to return (default: 2)")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    args = parser.parse_args()

    # Validate input
    if not args.file and not args.sample:
        print("Error: Please provide either a file path (-f) or use --sample")
        sys.exit(1)

    # Initialize matcher with correct file and columns
    if args.sample:
        print("Creating and loading sample data...")
        sample_file = "sample_demo_database.csv"
        matcher = DemoMatcher(sample_file, match_columns=["Problem/Need", "Solution Provided"])
        matcher.create_sample_data(sample_file)
    else:
        print(f"Loading data from {args.file}...")
        matcher = DemoMatcher(args.file, match_columns=["Client Problem", "Instalily AI Capabilities", "Benefit to Client"])

    if matcher.demos_df is None or matcher.demos_df.empty:
        print("Error: No data loaded!")
        sys.exit(1)

    # Search for similar demos
    print(f"\nSearching for demos similar to: '{args.customer_need}'")
    print("-" * 60)
    analysis = matcher.get_detailed_analysis(args.customer_need, args.top_k)
    print(analysis)

if __name__ == "__main__":
    main()
