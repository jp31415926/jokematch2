#!/usr/bin/env python3
"""
Simple test for TF-IDF search functionality that validates the script runs.
"""

import subprocess
import sys
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tfidf_search_basic():
    """Basic test to ensure the script runs without crashing."""
    # Check if artifacts files exist 
    required_files = [
        "tfidf_vectorizer.pkl",
        "tfidf_matrix.npz", 
        "tfidf_ids.pkl",
        "tfidf_titles.pkl"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            logger.warning(f"Artifact file not found: {file}")
            return False
    
    # Create a simple joke file
    joke_text = "Why don't scientists trust atoms?\nBecause they make up everything!"
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(joke_text)
        temp_file = f.name
    
    try:
        # Run search script
        result = subprocess.run([
            sys.executable, 
            "search_tfidf.py", 
            temp_file
        ], capture_output=True, text=True, check=False)

        logger.info(result.stdout)

        # Check if it ran successfully (exit code 0 means success)
        if result.returncode == 0:
            logger.info("TF-IDF search script ran successfully")
            # Check that we got output with the right format
            if "Rank   Score       ID     Title" in result.stdout:
                assert True, "Script produced expected output format"
                #return True
            else:
                logger.info(f"Output was: {result.stdout[:200]}")
                assert False, "Script didn't produce expected output format"
                #return False
        else:
            logger.error(f"Error output: {result.stderr}")
            assert False, f"Script failed with exit code {result.returncode}"
            #return False
            
    finally:
        # Clean up temp file
        Path(temp_file).unlink()

if __name__ == "__main__":
    success = test_tfidf_search_basic()
    if success:
        print("Basic TF-IDF search test PASSED")
    else:
        print("Basic TF-IDF search test FAILED")
        sys.exit(1)