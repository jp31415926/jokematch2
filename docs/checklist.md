# Project Development Checklist

 Make sure you include all new code.

To address the task, I'll provide a step-by-step explanation and the corresponding code modifications.
This checklist outlines all tasks needed to complete the **Joke Duplicate Detection Suite** as specified in `spec.md` and outlined in `prompt_plan.md`.

## Phase 0: Environment Setup

- [x] Create Python virtual environment
  - `python3 -m venv .venv`
- [x] Activate virtual environment
  - `source .venv/bin/activate`
- [x] Install dependencies
  - `pip install mysql-connector-python scikit-learn scipy joblib numpy sentence-transformers torch tqdm`
- [x] Create `requirements.txt`
  - `pip list | grep -E "(mysql-connector-python|scikit-learn|scipy|joblib|numpy|sentence-transformers|torch|tqdm)"`
  - Pin exact versions from installed packages
- [ ] Set up `.env` for DB credentials (optional)
  - Use `python-dotenv` if needed

## Phase 1: Database Layer (`db.py`)

- [x] Implement `db.py` with `fetch_jokes()`
  - Use `mysql.connector`, type hints, logging
- [x] Add `DBConnectionError` exception
  - Graceful error handling
- [x] Write `__main__` guard to print joke count
  - For testing
- [x] Create `tests/test_db.py`
  - Test that `fetch_jokes()` returns non-empty list
- [x] Run `pytest tests/test_db.py`
  - Confirm DB connection works

## Phase 2: TF-IDF Build Pipeline (`build_tfidf.py`)

- [x] Implement `build_tfidf.py`
  - Import `db.fetch_jokes()`
- [x] Instantiate `TfidfVectorizer` with spec settings
  - `stop_words='english'`, `lowercase=True`, `ngram_range=(1,2)`, `min_df=2`, `max_features=50000`
- [x] Fit vectorizer and transform texts
  - Save sparse matrix
- [x] Persist artifacts: `tfidf_vectorizer.pkl`, `tfidf_matrix.npz`, `tfidf_ids.pkl`, `tfidf_titles.pkl`
  - Use `joblib`, `scipy.sparse`, `pickle`
- [x] Print success message
  - "TF-IDF build finished: <n> jokes."
- [x] Create `tests/test_tfidf_build.py`
  - Test artifact existence, shape, idempotency
- [x] Run `pytest tests/test_tfidf_build.py`
  - Verify artifacts are created correctly

## Phase 3: TF-IDF Search Pipeline (`search_tfidf.py`)

- [ ] Implement `search_tfidf.py`
  - Accepts path to joke file
- [ ] Load persisted artifacts
  - Vectorizer, matrix, ids, titles
- [ ] Read input file and encode
  - Use `linear_kernel` for cosine similarity
- [ ] Rank top 10 matches
  - Use `np.argpartition`
- [ ] Print formatted table
  - Score, ID, Title
- [ ] Add error handling
  - Missing files, empty input
- [ ] Create `tests/test_tfidf_search.py`
  - Test output matches known joke
- [ ] Run `pytest tests/test_tfidf_search.py`
  - Validate correct ranking

## Phase 4: Transformer Build Pipeline (`build_tf.py`)

- [ ] Implement `build_tf.py`
  - Use `fetch_jokes()`
- [ ] Load `SentenceTransformer('all-MiniLM-L6-v2')`
  - Batch size = 64
- [ ] Encode all jokes
  - Save vectors as numpy array
- [ ] Persist artifacts: `tf_vectors.npy`, `tf_ids.npy`, `tf_titles.pkl`
  - Use `numpy.save`, `pickle`
- [ ] Print success message
  - "Transformer build finished: <n> jokes."
- [ ] Create `tests/test_tf_build.py`
  - Test artifact existence, shape
- [ ] Run `pytest tests/test_tf_build.py`
  - Confirm build works

## Phase 5: Transformer Search Pipeline (`search_tf.py`)

- [ ] Implement `search_tf.py`
  - Accepts path to joke file
- [ ] Load artifacts: vectors, ids, titles
- [ ] Load transformer model
- [ ] Encode query joke
- [ ] Compute cosine similarity
  - Dot product / norms
- [ ] Rank and print top 10
  - Same format as TF-IDF
- [ ] Add error handling
  - Missing files, model loading
- [ ] Create `tests/test_tf_search.py`
  - Test output matches known joke
- [ ] Run `pytest tests/test_tf_search.py`
  - Validate correct ranking

## Phase 6: Unified CLI Wrapper (`cli.py`)

- [ ] Implement `cli.py` with `argparse`
  - Subcommands: `build`, `search`
- [ ] Support flags: `--tfidf`, `--tf`, `--top N`, `--threshold FLOAT`
- [ ] Route to correct build or search logic
- [ ] Add `--version` flag
- [ ] Ensure `python cli.py` runs CLI
- [ ] Create `tests/test_cli.py`
  - Test build and search commands
- [ ] Run `pytest tests/test_cli.py`
  - Confirm CLI works

## Phase 7: Packaging & Distribution

- [ ] Create `setup.cfg`
  - Metadata, dependencies, entry points
- [ ] Create `MANIFEST.in`
  - Include artifact files
- [ ] Create `Dockerfile`
  - Use slim Python image, install deps
- [ ] Build source distribution
  - `python -m build`
- [ ] Build Docker image
  - `docker build -t joke-duplicate-detection .`
- [ ] Test CLI via Docker
  - Ensure it works inside container

## Phase 8: Integration Testing

- [ ] Create `tests/test_integration.py`
  - Full workflow test
- [ ] Run TF-IDF build → search → Transformer build → search
- [ ] Assert top results match expected joke
- [ ] Validate scores are within expected range
- [ ] Clean up temporary files
- [ ] Skip if DB/artifacts missing
- [ ] Run `pytest tests/test_integration.py`
  - End-to-end validation

## Phase 9: Final Documentation & Quality Assurance

- [ ] Write `README.md`
  - Description, quickstart, usage, testing
- [ ] Generate `requirements.txt`
  - Pin exact versions
- [ ] Run full test suite
  - `pytest` with coverage
- [ ] Review performance
  - Memory, speed
- [ ] Add contribution guidelines
  - For future maintainers
- [ ] Final code review
  - Ensure best practices followed

## 📦 Final Deliverables

- [ ] `db.py`
- [ ] `build_tfidf.py`
- [ ] `search_tfidf.py`
- [ ] `build_tf.py`
- [ ] `search_tf.py`
- [ ] `cli.py`
- [ ] `setup.cfg`
- [ ] `MANIFEST.in`
- [ ] `Dockerfile`
- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] All unit and integration tests

## 🧪 Testing Strategy

- [ ] Unit Tests (each module)
- [ ] Integration Tests (end-to-end)
- [ ] CLI Tests
- [ ] Performance Benchmarks
- [ ] Coverage Reports

> 💡 **Tip:** Use `pytest --maxfail=1` during development to catch failures early.

> 🔄 **Tip:** Re-run tests after each chunk to ensure no regressions.