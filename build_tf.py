#!/usr/bin/env python3
"""
Transformer build script for joke duplicate detection suite.
"""
import logging
import sys
import os
import numpy as np
import pickle
from typing import List, Tuple, Any
from pathlib import Path

# Add the project root to the path so we can import from db module
sys.path.insert(0, str(Path(__file__).parent))

from db import fetch_jokes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_transformer_pipeline() -> int:
    """Build transformer pipeline for joke vectorization."""
    try:
        # Fetch jokes from the database
        logger.info("Fetching jokes from database...")
        jokes = fetch_jokes()
        
        if not jokes:
            logger.warning("No jokes found in database")
            return 0
            
        # Extract ids, titles, and texts from the list of tuples
        ids, titles, texts = zip(*jokes)  # type: ignore
        
        # Load the sentence transformer model
        logger.info("Loading sentence transformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode all jokes in batches
        batch_size = 128
        logger.info(f"Encoding {len(texts)} jokes in batches of {batch_size}...")
        vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vectors = model.encode(batch_texts)  # type: ignore
            vectors.append(batch_vectors)
        
        # Concatenate all vectors
        vectors = np.vstack(vectors)
        
        # Validate the shape
        assert vectors.shape == (len(ids), 384), f"Unexpected vector shape: {vectors.shape}"
        
        # Persist the artifacts
        logger.info("Persisting artifacts...")
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)

        # Save vectors
        np.save(output_dir / "tf_vectors.npy", vectors)
        
        # Save ids
        np.save(output_dir / "tf_ids.npy", np.array(ids))
        
        # Save titles
        title_map = dict(zip(ids, titles))
        with open(output_dir / "tf_titles.pkl", "wb") as f:
            pickle.dump(title_map, f)
        
        logger.info(f"Transformer build finished: {len(ids)} jokes.")
        return len(ids)
        
    except Exception as e:
        logger.error(f"Error in transformer build pipeline: {e}")
        raise

if __name__ == "__main__":
    try:
        build_transformer_pipeline()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)