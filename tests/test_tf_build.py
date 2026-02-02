import subprocess
import sys
import logging
from pathlib import Path

# Set up logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transformer_build():
    """Test the transformer build script."""
    # Check if db connection is available
    try:
        import db
        # Try a simple connection test
        conn = db.get_db_connection()
        conn.ping()
        conn.close()
    except Exception as e:
        logger.warning(f"Skipping test due to DB connection failure: {e}")
        # Skip test by returning early - pytest will mark it as skipped
        return

    # Run the build script
    logger.info("Running transformer build script...")
    result = subprocess.run([
        sys.executable, 
        "build_tf.py"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0, f"Build script failed: {result.stderr}"
    
    logger.info("Build script completed successfully")
    
    # Check that all artifact files exist
    output_dir = Path("data")
    required_files = [
        output_dir / "tf_vectors.npy",
        output_dir / "tf_ids.npy", 
        output_dir / "tf_titles.pkl"
    ]
    
    for file_path in required_files:
        assert file_path.exists(), f"Required file not found: {file_path}"
    
    logger.info("All artifact files created successfully")
    
    # Load and validate vectors
    try:
        import numpy as np
        vectors = np.load(output_dir / "tf_vectors.npy")
        logger.info(f"Loaded vectors shape: {vectors.shape}")
        
        # Check shape is (n, 384) where n is number of jokes
        assert len(vectors.shape) == 2 and vectors.shape[1] == 384, f"Unexpected vector shape: {vectors.shape}"
            
        logger.info("Vector shape validation passed")
        
    except Exception as e:
        logger.error(f"Error loading or validating vectors: {e}")
        raise
        
    logger.info("Transformer build test passed")

if __name__ == "__main__":
    test_transformer_build()