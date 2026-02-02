#!/usr/bin/env python3
"""
Test for transformer search functionality.
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

def test_tf_search():
    """Test the transformer search script."""
    # Check if artifacts files exist (they should be in the data directory)
    artifact_dir = Path("data")
    required_files = [
        artifact_dir / "tf_vectors.npy",
        artifact_dir / "tf_ids.npy", 
        artifact_dir / "tf_titles.pkl"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            logger.warning(f"Artifact file not found: {file}, skipping test")
            pytest.skip(f"Artifact file not found: {file}")
    
    # For this test, we will create a simple joke input and run the search on it
    # We'll try to use the first joke from the database as test input to get meaningful results
    
    # Try to access DB to get a sample joke
    # Import and test DB connection
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db import fetch_jokes
    
    jokes = fetch_jokes()
    if not jokes:
        logger.warning("No jokes found in DB, skipping test")
        pytest.skip("No jokes found in database")
        
    # Get the first joke from DB to use as input
    test_id, test_title, test_joke = jokes[1234]
    logger.info(f"Using joke from DB as test input: {test_joke[:50]}...")
    
    # Create temporary joke file with content from DB
    with tempfile.NamedTemporaryFile(mode="w", dir="data/", suffix=".txt", delete=False) as f:
        f.write(test_joke)
        temp_file = f.name

    try:
        # Run search script
        result = subprocess.run([
            sys.executable, 
            "search_tf.py", 
            temp_file
        ], capture_output=True, text=True, check=True)

        logger.info(result.stdout)

        # Parse output
        output_lines = result.stdout.strip().split('\n')
        
        # Check header line
        assert output_lines[0] == "Rank   Score       ID     Title"
        
        # Check that at least one result was returned
        assert len(output_lines) > 1, "Expected at least one result"
        
        # Check format of first result
        first_line = output_lines[1]
        parts = first_line.split()
        assert len(parts) >= 4, "Expected at least 4 columns in result"
        rank = int(parts[0])
        score = float(parts[1])
        returned_id = int(parts[2])
        assert rank > 0 and rank < 11, "Rank should between 1-10"
        assert score > 0.0, "Score should be positive"
        assert returned_id > 0, "ID should be positive"
        
        # Verify that the first result has a high score and correct ID for our test
        # Since we're comparing the same joke to itself, we expect a high score (close to 1.0)
        assert score >= 0.95, f"Expected score >= 0.95 for identical joke, got {score}"
        assert returned_id == test_id, f"Expected ID {test_id}, got {returned_id}"
        
        logger.info("Transformer search test passed")
        
    finally:
        # Clean up temp file
        Path(temp_file).unlink()

if __name__ == "__main__":
    test_tf_search()