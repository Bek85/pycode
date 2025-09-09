#!/usr/bin/env python3
"""
Test script to demonstrate colorful result output.
"""

import sys
import os

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_demos.utils.logging import colorize_result_output, colorize_execution_time, ColorCodes

def test_colorful_results():
    """Test colorful result output formatting."""
    
    print(f"{ColorCodes.BRIGHT_CYAN}=== Colorful Result Output Test ==={ColorCodes.RESET}")
    print()
    
    # Test various result formats
    results = [
        "There are **1,500 orders** in the database.",
        "There are **4,000 products** in the database.",
        "Total revenue is $123,456.89 for the month.",
        """Here are the 5 most recent orders:

| Order ID | Date | Amount |
|----------|------|---------|
| 754 | 2026-12-28 | $12,689.45 |
| 467 | 2026-12-27 | $31,865.88 |
| 1347 | 2026-12-26 | $9,618.30 |""",
        "Found 25 users with 150 total orders.",
        "Database contains 6 tables with 10,500 total records."
    ]
    
    execution_times = [5.2, 13.7, 0.8, 25.3, 2.1, 16.8]
    
    for i, result in enumerate(results):
        print(f"{ColorCodes.BRIGHT_GREEN}[Result #{i+1}]{ColorCodes.RESET}")
        print(colorize_result_output(result))
        print()
        print(colorize_execution_time(execution_times[i]))
        print()
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_colorful_results()