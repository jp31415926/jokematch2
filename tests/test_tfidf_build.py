#!/usr/bin/env python3
"""
Tests for the TF-IDF build script.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import joblib
from scipy import sparse

from db import fetch_jokes

# Configure logging to avoid warnings during test discovery
logging.basicConfig(level=logging.WARNING)


def test_tfidf_build():
    """Test that the TF-IDF build script creates expected files and works correctly."""
    # First, check if we can connect to the database
    try:
        jokes = fetch_jokes()
    except Exception:
        # Skip test if database connection fails
        pytest.skip("Database connection failed, skipping test")

    # Run the build script
    result = subprocess.run([
        sys.executable, "build_tfidf.py"
    ], capture_output=True, text=True, cwd=".")

    assert result.returncode == 0, f"Build script failed: {result.stderr}"

    # Check that all four artifact files were created
    output_dir = Path(".")
    assert (output_dir / "tfidf_vectorizer.pkl").exists()
    assert (output_dir / "tfidf_matrix.npz").exists()
    assert (output_dir / "tfidf_ids.pkl").exists()
    assert (output_dir / "tfidf_titles.pkl").exists()

    # Load the vectorizer and matrix to verify they are valid
    vectorizer = joblib.load(output_dir / "tfidf_vectorizer.pkl")
    tfidf_matrix = sparse.load_npz(output_dir / "tfidf_matrix.npz")

    # Check matrix shape matches number of jokes
    assert tfidf_matrix.shape[0] == len(jokes)
    # Check the number of features
    assert tfidf_matrix.shape[1] == len(vectorizer.vocabulary_)

    # Verify that we have the right number of joke IDs
    with open(output_dir / "tfidf_ids.pkl", "rb") as f:
        ids = joblib.load(f)
    assert len(ids) == len(jokes)

    # Verify that the titles dictionary has the right number of entries
    with open(output_dir / "tfidf_titles.pkl", "rb") as f:
        titles = joblib.load(f)
    assert len(titles) == len(jokes)

    # Check that all IDs from database are in the saved IDs
    db_ids = {joke[0] for joke in jokes}
    saved_ids = set(ids)
    assert db_ids.issubset(saved_ids)