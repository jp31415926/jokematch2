#!/usr/bin/env python3
"""
TF-IDF search script for joke duplicate detection.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_artifacts(artifacts_dir: Path) -> Tuple:
    """Load persisted TF-IDF artifacts."""
    try:
        vectorizer_path = artifacts_dir / "tfidf_vectorizer.pkl"
        matrix_path = artifacts_dir / "tfidf_matrix.npz"
        ids_path = artifacts_dir / "tfidf_ids.pkl"
        titles_path = artifacts_dir / "tfidf_titles.pkl"
        
        # Load vectorizer (saved with joblib)
        vectorizer = joblib.load(vectorizer_path)
        
        # Load sparse matrix properly
        tfidf_matrix = load_npz(matrix_path)
        
        # Load IDs (saved with pickle)
        import pickle
        with open(ids_path, "rb") as f:
            joke_ids = pickle.load(f)
        
        # Load titles (saved with pickle)
        with open(titles_path, "rb") as f:
            joke_titles = pickle.load(f)
            
        logger.info("Successfully loaded TF-IDF artifacts")
        return vectorizer, tfidf_matrix, joke_ids, joke_titles
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise

def search_joke(query_text: str, vectorizer, tfidf_matrix, joke_ids, joke_titles) -> List[Tuple[float, int, str]]:
    """Search for similar jokes using TF-IDF and cosine similarity."""
    try:
        logger.info(f"len(joke_ids) = {len(joke_ids)}")
        logger.info(f"len(joke_titles) = {len(joke_titles)}")

        # Transform query text to TF-IDF vector
        query_vector = vectorizer.transform([query_text])
        logger.info(f"query_vector calculated")
        
        # Compute cosine similarities
        similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
        logger.info(f"similarities calculated")
        
        # Get top 10 most similar jokes (descending order)
        top_indices = similarities.argsort()[::-1][:10]
        logger.info(f"Found {len(top_indices)} similar jokes")
        
        # Format results - make sure we handle types properly
        results = []
        for i in top_indices:
            logger.info(f"i = {i}")
            score = float(similarities[i])
            logger.info(f"score = {score}")
            joke_id = int(joke_ids[i])  # Properly convert to int
            logger.info(f"joke_id = {joke_id}")
            logger.info(f"joke_titles[i] = {len(joke_titles[i] if joke_titles[i] is not None else "")}")
            title = str(joke_titles[i]) if joke_titles[i] is not None else ""
            logger.info(f"title = {title}")
            results.append((score, joke_id, title))
            logger.info(f"results now contains {len(results)} results")
            
        logger.info(f"Found {len(results)} similar jokes")
        return results
        
    except Exception as e:
        logger.error(f"Error during search: {repr(e)}")
        raise

def main():
    """Main function to handle CLI arguments and execute search."""
    parser = argparse.ArgumentParser(description="Search for similar jokes using TF-IDF")
    parser.add_argument("joke_file", help="Path to a plain-text file containing a new joke")
    
    args = parser.parse_args()
    
    try:
        # Find artifacts directory - look in project root
        script_dir = Path(__file__).parent
        artifacts_dir = script_dir  # Changed from script_dir / "artifacts" to just script_dir
        
        logger.info(f"Loading artifacts from {artifacts_dir}")
        vectorizer, tfidf_matrix, joke_ids, joke_titles = load_artifacts(artifacts_dir)
        
        # Read joke file
        joke_file_path = Path(args.joke_file)
        if not joke_file_path.exists():
            logger.error(f"Joke file not found: {joke_file_path}")
            sys.exit(1)
            
        with open(joke_file_path, "r", encoding="utf-8") as f:
            joke_text = f.read().strip()
            
        if not joke_text:
            logger.error("Input joke file is empty")
            sys.exit(1)
            
        logger.info(f"Searching for: {joke_text[:50]}...")
        
        # Perform search
        results = search_joke(joke_text, vectorizer, tfidf_matrix, joke_ids, joke_titles)
        
        # Print results table
        print("score   id   title")
        for score, joke_id, title in results:
            print(f"{score:.4f}   {joke_id}   {title}")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()