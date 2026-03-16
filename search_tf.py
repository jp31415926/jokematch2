#!/usr/bin/env python3
"""
Transformer search script for joke duplicate detection.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_artifacts(artifacts_dir: Path) -> Tuple:
    """Load persisted transformer artifacts."""
    try:
        vectors_path = artifacts_dir / "tf_vectors.npy"
        ids_path = artifacts_dir / "tf_ids.npy"
        titles_path = artifacts_dir / "tf_titles.pkl"
        
        # Load vectors
        vectors = np.load(vectors_path)
        
        # Load IDs
        ids = np.load(ids_path)
        
        # Load titles
        with open(titles_path, "rb") as f:
            titles = pickle.load(f)
            
        logger.info("Successfully loaded transformer artifacts")
        return vectors, ids, titles
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise

def search_joke(query_text: str, vectors, ids, titles) -> List[Tuple[float, int, str]]:
    """Search for similar jokes using transformer vectors and cosine similarity."""
    try:
        # Load the sentence transformer model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode query joke
        logger.info("Encoding query joke...")
        query_vec = model.encode([query_text], show_progress_bar=False)[0]
        
        # Compute cosine similarity: dot(vectors, query_vec) / (norm(vectors, axis=1) * norm(query_vec))
        # This matches the implementation described in the spec
        similarities = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec))
        
        # Get top 10 most similar jokes (descending order)
        top_indices = np.argpartition(similarities, -10)[-10:][::-1]
        
        # Format results
        results = []
        for i in top_indices:
            score = float(similarities[i])
            joke_id = int(ids[i])
            title = str(titles.get(joke_id, "")) if titles.get(joke_id) is not None else ""
            results.append((score, joke_id, title))
            
        return results
        
    except Exception as e:
        logger.error(f"Error during search: {repr(e)}")
        raise

def main():
    """Main function to handle CLI arguments and execute search."""
    parser = argparse.ArgumentParser(description="Search for similar jokes using transformer embeddings")
    parser.add_argument("joke_file", help="Path to a plain-text file containing a new joke")
    
    args = parser.parse_args()
    
    try:
        artifacts_dir = Path("data")  # artifacts located in data directory
        
        logger.info(f"Loading transformer artifacts from {artifacts_dir}")
        vectors, ids, titles = load_artifacts(artifacts_dir)
        
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
        results = search_joke(joke_text, vectors, ids, titles)
        
        # Print results table
        print(f"{'Rank':<6} {'Score':<10} {'ID':^5}   Title")
        for rank, (score, joke_id, title) in enumerate(results, start=1):
            print(f"{rank:>2}     {score:<10.4f} {joke_id:>5}   {title:<40}")

    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()