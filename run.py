#!/usr/bin/env python3
"""
Simple execution script for the Food AI Ingredient Scaling System.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

def main():
    """Run the main application with default demo settings."""
    
    # Check if data file exists
    data_file = current_dir / 'data' / 'paneer_recipes.json'
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure paneer_recipes.json is in the data/ directory")
        return 1
    
    # Use virtual environment Python if available
    venv_python = current_dir / '.venv' / 'bin' / 'python'
    python_exec = str(venv_python) if venv_python.exists() else 'python3'
    
    # Default arguments for demo (simplified)
    args = [python_exec, 'quick_demo.py']
    
    print("Running Food AI Ingredient Scaling System Demo...")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Output directory: {current_dir / 'output'}")
    print("="*60)
    
    try:
        # Run the demo script
        result = subprocess.run(args, cwd=current_dir, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1

if __name__ == "__main__":
    exit(main())
