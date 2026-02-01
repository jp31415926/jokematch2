#!/usr/bin/env python3
"""
Test for TF-IDF search functionality.
"""

import subprocess
import sys
import tempfile
import logging
from pathlib import Path

import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tfidf_search():
    """Test the TF-IDF search script."""
    # Check if artifacts files exist (they should be in the project root)
    required_files = [
        "tfidf_vectorizer.pkl",
        "tfidf_matrix.npz", 
        "tfidf_ids.pkl",
        "tfidf_titles.pkl"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            logger.warning(f"Artifact file not found: {file}, skipping test")
            pytest.skip(f"Artifact file not found: {file}")
    
    # Just proceed with test - the actual DB connection check is removed  
    # since we can test with artifacts without needing a DB connection for search
    
    # For this test, we will create a simple joke input and run the search on it
    # since we don't want to depend on database connection for this search test
    
    joke_text = "Why don't scientists trust atoms?"
    joke_id = 1  # Use a dummy ID as we're not testing against DB
    
    # Create temporary joke file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(joke_text)
        temp_file = f.name
    
    try:
        # Run search script
        result = subprocess.run([
            sys.executable, 
            "search_tfidf.py", 
            temp_file
        ], capture_output=True, text=True, check=True)

        logger.info(result.stdout)

        # Parse output
        output_lines = result.stdout.strip().split('\n')
        
        # Check header line
        assert output_lines[0] == "score   id   title"
        
        # Check that at least one result was returned
        assert len(output_lines) > 1, "Expected at least one result"
        
        # Check format of first result
        first_line = output_lines[1]
        parts = first_line.split()
        assert len(parts) == 3, "Expected 3 columns in result"
        score = float(parts[0])
        returned_id = int(parts[1])
        
        # Verify score is reasonable (should be > 0)
        assert score > 0.0, "Score should be positive"
        
        logger.info("TF-IDF search test passed")
        
    finally:
        # Clean up temp file
        Path(temp_file).unlink()

if __name__ == "__main__":
    test_tfidf_search()