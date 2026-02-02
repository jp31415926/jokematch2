# 1.  Detailed, Step‑by‑Step Blueprint

| Phase | Goal | Deliverable | Key Actions |
|-------|------|-------------|-------------|
| **0.  Environment** | Set up a clean dev environment | `venv`, `requirements.txt` | `python3 -m venv .venv`<br>`source .venv/bin/activate`<br>`pip install -r requirements.txt` |
| **1.  DB Layer** | Abstract MySQL access | `db.py` | Write a function `fetch_jokes()` that returns a list of tuples `(id, title, funny)` |
| **2.  TF‑IDF Build** | Compute TF‑IDF matrix | `tfidf_vectorizer.pkl`, `tfidf_matrix.npz`, `tfidf_ids.pkl`, `tfidf_titles.pkl` | Pull data from DB → vectorize → save |
| **3.  TF‑IDF Search** | Rank a new joke | `search_tfidf.py` | Load artifacts → encode query → cosine similarity → print top‑10 |
| **4.  Transformer Build** | Compute sentence‑transformer vectors | `tf_vectors.npy`, `tf_ids.npy`, `tf_titles.pkl` | Pull data from DB → encode → save |
| **5.  Transformer Search** | Rank a new joke | `search_tf.py` | Load artifacts → encode query → cosine similarity → print top‑10 |
| **6.  CLI & Packaging** | Provide unified CLI & packaging | `cli.py`, `setup.cfg`, `MANIFEST.in` | Wrap all scripts into a single entry‑point, add command‑line options |
| **7.  Testing** | Verify every component | `tests/` | Write unit & integration tests that use the real DB and real calls |
| **8.  Deployment** | Run nightly builds & searches | cron, Dockerfile | Schedule `build_*` jobs; run `search_*` on new emails |

---

# 2.  Iterative Chunks (Building Blocks)

| Chunk | Target Artifacts | Why it’s a Chunk | Dependencies |
|-------|------------------|------------------|--------------|
| **A. DB Connector** | `db.py` | Provides a single source of truth for DB access. | None |
| **B. TF‑IDF Build** | `build_tfidf.py`, artifacts | First heavy computation; must be idempotent. | `db.py` |
| **C. TF‑IDF Search** | `search_tfidf.py` | Stateless lookup; reads artifacts. | `build_tfidf.py` artifacts |
| **D. Transformer Build** | `build_tf.py`, artifacts | Second heavy computation; independent. | `db.py` |
| **E. Transformer Search** | `search_tf.py` | Stateless lookup; reads artifacts. | `build_tf.py` artifacts |
| **F. CLI Wrapper** | `cli.py` | Unified entry‑point; optional flags. | A–E |
| **G. Test Suite** | `tests/` | Validates correctness. | A–F |
| **H. Packaging** | `setup.cfg`, `MANIFEST.in`, Dockerfile | Distribution & reproducibility. | G |

> Each chunk is small enough that a single commit can implement it, yet large enough to move the project forward.

---

# 3.  Step‑by‑Step Inside Each Chunk

## 3.1  Chunk A – DB Connector (`db.py`)

1. Import `mysql.connector` and `typing`.  
2. Define `DB_CONFIG` (same as in the spec).  
3. Write `fetch_jokes() -> List[Tuple[int, str, str]]`.  
4. Add `__main__` guard to print a sample row count.  

## 3.2  Chunk B – TF‑IDF Build

1. Import `build_tfidf.py` → `db.fetch_jokes()`.  
2. Split rows into `ids`, `titles`, `texts`.  
3. Instantiate `TfidfVectorizer` with settings from the spec.  
4. Fit/transform `texts`.  
5. Persist `vectorizer` with `joblib`, matrix with `scipy.sparse.save_npz`, and meta with `pickle`.  
6. Print a success message.  

## 3.3  Chunk C – TF‑IDF Search

1. Load vectorizer, matrix, ids, titles.  
2. Read query file from `sys.argv[1]`.  
3. Encode query.  
4. Compute cosine similarity via `linear_kernel`.  
5. Select top‑10 indices with `np.argpartition`.  
6. Format and print results.  

## 3.4  Chunk D – Transformer Build

1. Load jokes via `db.fetch_jokes()`.  
2. Encode with `SentenceTransformer('all-MiniLM-L6-v2')`.  
3. Persist vectors (`np.save`), ids (`np.save`), titles (`pickle`).  

## 3.5  Chunk E – Transformer Search

1. Load vectors, ids, titles.  
2. Read query, encode with same transformer.  
3. Compute cosine similarity (dot / norms).  
4. Rank and output.  

## 3.6  Chunk F – CLI Wrapper

1. Use `argparse` to expose `--tfidf` and `--tf` sub‑commands.  
2. Wire each sub‑command to its corresponding build or search function.  
3. Add optional `--threshold`, `--top` arguments.  

## 3.7  Chunk G – Test Suite

1. `tests/test_db.py` – ensure `fetch_jokes()` returns non‑empty list.  
2. `tests/test_tfidf_build.py` – run `build_tfidf.py`, assert artifacts exist, shapes.  
3. `tests/test_tfidf_search.py` – run `search_tfidf.py` on a sample joke, assert output contains known IDs.  
4. Similar tests for transformer.  
5. Integration test: run both builds, then searches, compare top‑10 lists.  

## 3.8  Chunk H – Packaging

1. `setup.cfg` with metadata, `entry_points` for `joke_search`.  
2. `MANIFEST.in` to include artifact files.  
3. `Dockerfile` that installs deps, copies code, exposes CLI.  

---

# 4.  Prompts for a Code‑Generation LLM

> **Guidelines for each prompt**  
> • Provide full context (the spec, the previous step).  
> • Instruct the LLM to write **Python 3.11+** code.  
> • Ask for **unit tests** that use real data (connect to the DB, read actual files).  
> • Emphasize best practices: type hints, logging, `__main__` guard, idempotence.  
> • Each prompt should reference the *previous* one so that the code builds incrementally.

Below are the prompts, each wrapped in Markdown code fences.  
Copy them one by one into your LLM; it will generate the requested code and tests.

---

## Prompt 1 – DB Connector (`db.py`)

```text
You are a Python developer tasked with creating a lightweight MySQL connector module for a joke duplicate‑detection suite.  
The module must provide a single function `fetch_jokes()` that returns a list of tuples `(id, title, funny)` from the `archives` table.  
Use the database configuration provided in the specification, but put this in it's own file called `db_config.py`.

DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_user',
    'password': 'your_password',
    'database': 'your_database',
}

Requirements:
- Use `mysql.connector` (the official MySQL driver).
- Include proper type hints and docstrings.
- Handle connection errors gracefully: log an error and raise a custom `DBConnectionError`.
- Write a `__main__` guard that prints the number of jokes fetched.
- Ensure that the function is idempotent and can be safely called multiple times.
- Add a small test suite in a separate file `tests/test_db.py` that:
  - Calls `fetch_jokes()` and asserts the return type is `List[Tuple[int, str, str]]`.
  - Checks that the list is non‑empty (i.e., at least one joke exists).
  - If the database is not configured, the test should be skipped with an informative message.

Use Python 3.11+ syntax, standard library only for the test (you may use `pytest` for the test file).  
Return the code for `db.py` and the test file `tests/test_db.py`.  
Do NOT write any code for other modules yet.
```

---

## Prompt 2 – TF‑IDF Build Script (`build_tfidf.py`)

```text
cContinuing from the previous `db.py` module, now write a script `build_tfidf.py` that:
1. Imports `fetch_jokes` from `db.py`.
2. Pulls all rows: `ids`, `titles` (dict mapping id → title), and `texts` (list of funny jokes).
3. Builds a `TfidfVectorizer` with these settings:
   - stop_words='english'
   - lowercase=True
   - ngram_range=(1,2)
   - min_df=2
   - max_features=50000
4. Fits the vectorizer on `texts` and obtains a sparse CSR matrix.
5. Persists the following artifacts to the current directory:
   - `tfidf_vectorizer.pkl` (vectorizer via `joblib.dump`)
   - `tfidf_matrix.npz` (matrix via `scipy.sparse.save_npz`)
   - `tfidf_ids.pkl` (pickle of list `ids`)
   - `tfidf_titles.pkl` (pickle of dict `titles`)
6. Prints a message: “TF‑IDF build finished: <n> jokes.”  
   Ensure idempotency: re‑running overwrites the files.
7. Add a test `tests/test_tfidf_build.py` that:
   - Runs the build script via `subprocess.run`.
   - Asserts that all four artifact files exist.
   - Loads the vectorizer and matrix; checks that the matrix shape matches the number of jokes and vectorizer’s feature count.
   - Skips the test if the database connection fails.

Use Python 3.11+ syntax, `typing`, `logging`.
Return only the code for `build_tfidf.py` and `tests/test_tfidf_build.py`.
Create these files and place them where they should go in the project.
Do NOT create other modules yet.
```

---

## Prompt 3 – TF‑IDF Search Script (`search_tfidf.py`)

```text
You are a Python developer tasked with creating a lightweight MySQL connector module for a joke duplicate‑detection suite.  
Using the artifacts from the previous build, write a search script `search_tfidf.py` that:
1. Accepts a single positional argument: path to a plain‑text file containing a new joke.
2. Loads the persisted TF‑IDF vectorizer, matrix, ids, and titles.
3. Reads the joke file, encodes it into a TF‑IDF vector.
4. Computes cosine similarity between the query vector and all stored vectors using `sklearn.metrics.pairwise.linear_kernel`.
5. Retrieves the top‑10 most similar jokes (descending similarity).
6. Prints a table to stdout with columns: score (4 decimals), id, title. Header line: “score   id   title”.
7. Handles errors gracefully: missing files, empty input, etc., with clear messages.

Add a test `tests/test_tfidf_search.py` that:
- Creates a temporary joke file containing the exact text of the first joke from the database.
- Runs `search_tfidf.py` on that file via `subprocess.run`.
- Parses the stdout and verifies that the first result has a score close to 1.0 (within 0.01) and that its id matches the expected id.
- Skips the test if artifacts are missing or the DB is not reachable.

Use Python 3.11+ syntax, type hints, `logging`.  
Return only the code for `search_tfidf.py` and `tests/test_tfidf_search.py`.  
Create these files and place them where they should go in the project.
Do NOT write any other modules yet.
```

---

## Prompt 4 – Transformer Build Script (`build_tf.py`)

```text
You are a Python developer tasked with creating a lightweight MySQL connector module for a joke duplicate‑detection suite. Read `docs/spec.md` and `docs/AGENTS.md`!
Now build the second pipeline (Phase 5). Write `build_tf.py` that:
1. Uses `fetch_jokes` from `db.py` to obtain `ids`, `titles`, `texts`.
2. Loads `sentence_transformers.SentenceTransformer` with the model name `all-MiniLM-L6-v2`.
3. Encodes all jokes in batches (batch_size=64) into 384‑dimensional vectors.
4. Persists:
   - `tf_vectors.npy` (numpy array of shape (n, 384))
   - `tf_ids.npy` (numpy array of ints)
   - `tf_titles.pkl` (pickle of dict mapping id → title)
5. Prints “Transformer build finished: <n> jokes.”
6. Ensures idempotency.

Add a test `tests/test_tf_build.py` that:
- Runs the build script via `subprocess.run`.
- Checks that all three artifact files exist.
- Loads the vectors and verifies shape: (n, 384).
- Skips with informative message if DB connection fails.

The current directory is `/mnt/c/work/jokematch2/`. This is the base of the project. All filenames are relative to that path if they don't begin with a `/` character.
Use Python 3.11+, type hints, `logging`.
Return only the code for `build_tf.py` and `tests/test_tf_build.py`.  
Create these files and place them where they should go in the project.
Test both scripts to ensure they run without errors. If they don’t, iterate on fixes and retest.
Do NOT create other modules/scripts yet.
```

---

## Prompt 5 – Transformer Search Script (`search_tf.py`)

```text
You are a Python developer tasked with creating a lightweight MySQL connector module for a joke duplicate‑detection suite. Read `docs/spec.md` and `docs/AGENTS.md`!
Write `search_tf.py` that:
1. Accepts a path to a plain‑text joke file.
2. Loads `data/tf_vectors.npy`, `data/tf_ids.npy`, `data/tf_titles.pkl`.
3. Loads the same transformer model (`all-MiniLM-L6-v2`).
4. Encodes the query joke.
5. Computes cosine similarity: `dot(vectors, query_vec) / (norm(vectors, axis=1) * norm(query_vec))`.
6. Retrieves top‑10 most similar jokes.
7. Prints a table exactly like the TF‑IDF search. Reference `docs/spec.md` and `search_tfidf.py`.

Add a test `tests/test_tf_search.py` that:
- Creates a temporary joke file identical to the first DB joke.
- Runs `search_tf.py` via subprocess.
- Parses stdout; verifies the first result’s score is ≥0.95 and id matches the expected id.
- Skips if artifacts missing or DB not reachable.

The current directory is `/mnt/c/work/jokematch2/`. This is the base of the project. All filenames are relative to that path if they don't begin with a `/` character.
Use Python 3.11+, type hints, `logging`.  
Return only the code for `search_tf.py` and `tests/test_tf_search.py`.  
Create these files and place them where they should go in the project.
Test both scripts to ensure they run without errors. If they don’t, iterate on fixes and retest.
Do NOT create the CLI wrapper yet.
```

---

## Prompt 6 – Unified CLI Wrapper (`cli.py`)

```text
You are a Python developer tasked with creating a lightweight MySQL connector module for a joke duplicate‑detection suite.  
Create a single entry‑point script `cli.py` that:
1. Uses `argparse` to provide two mutually exclusive sub‑commands:
   - `build --tfidf` → runs `build_tfidf.py` logic (you can import the function from that module if you refactor it).
   - `build --tf`   → runs `build_tf.py` logic.
   - `search --tfidf <file>`
   - `search --tf <file>`
2. Optionally accepts `--top N` (default 10) and `--threshold FLOAT` (default 0.0) to filter the results.
3. For each search command, prints the same table format as the individual scripts.
4. Handles missing artifacts or DB errors gracefully with user‑friendly messages.
5. Adds a `--version` flag showing the tool version.
6. Ensures that running the script as `python cli.py` executes the CLI.

Add a test `tests/test_cli.py` that:
- Runs `python cli.py build --tfidf` and checks the artifact file exists.
- Runs `python cli.py search --tfidf <path_to_sample_joke>` and verifies the stdout contains at least one line with a numeric score.
- Skips if artifacts missing or DB not reachable.

Use Python 3.11+, type hints, `logging`.  
Return only the code for `cli.py` and `tests/test_cli.py`.  
Do NOT write packaging files yet.
```

---

## Prompt 7 – Packaging Configuration (`setup.cfg`, `MANIFEST.in`, `Dockerfile`)

```text
Prepare packaging and distribution for the tool.

1. `setup.cfg`:
   - Name: joke-duplicate-detection
   - Version: 0.1.0
   - Author, email, description.
   - Dependencies: listed in the spec (mysql-connector-python, scikit-learn, scipy, joblib, numpy, sentence-transformers, torch, tqdm).
   - entry_points: console_scripts: joke-search=cli:main
   - include_package_data = True
   - packages = find:

2. `MANIFEST.in`:
   - Include all .pkl, .npz, .npy artifacts when building a source distribution.

3. `Dockerfile`:
   - Use `python:3.11-slim-bookworm` as base.
   - Install `gcc`, `git`, `libopenblas-dev`, `libssl-dev`, `libffi-dev`.
   - Copy the project directory.
   - Run `pip install --no-cache-dir -r requirements.txt` (generate requirements.txt from `setup.cfg` or use `pip freeze`).
   - Set entrypoint to the CLI script.
   - Expose no ports; it's a command‑line tool.

Return the contents of `setup.cfg`, `MANIFEST.in`, and the Dockerfile.  
Do NOT alter any existing Python modules.  
```

---

## Prompt 8 – Final Integration Test (`tests/test_integration.py`)

```text
Write an integration test that verifies the entire workflow:

1. It will run the TF‑IDF build, then run a search on a sample joke, then run the Transformer build, and run a search.
2. It should use the same sample joke file created from the first DB joke.
3. The test asserts that:
   - Both searches return at least one result.
   - The top result from each pipeline shares the same id (the exact first joke).
   - Scores are within a reasonable range (TF‑IDF ≥0.8, Transformer ≥0.9).
4. It cleans up any temporary files created.
5. Skip the test if DB not reachable or artifacts missing.

Use `pytest`, `subprocess`, and `tempfile`.  
Return only the code for `tests/test_integration.py`.  
```

---

## Prompt 9 – Final Quality‑Check and Documentation (`README.md`, `requirements.txt`)

```text
Produce a comprehensive README that includes:
- Project description and use‑cases.
- Quick start: environment setup, running builds, searching.
- CLI usage examples with flags.
- Testing instructions (pytest, coverage).
- Performance notes (memory, speed).
- Contribution guidelines.

Generate a `requirements.txt` that pins the exact versions of all dependencies used by the project (you may use `pip freeze > requirements.txt` as a starting point).

Return the content for `README.md` and `requirements.txt`.  
```

---

### How to Use These Prompts

1. **Start with Prompt 1** – generate `db.py` and its tests.  
2. **Run the tests** (`pytest tests/test_db.py`). Ensure the database is reachable.  
3. **Proceed to Prompt 2** – generate the TF‑IDF build script and its test.  
4. **Run the build test** (`pytest tests/test_tfidf_build.py`). Verify artifacts are created.  
5. **Continue** through Prompts 3–8, each time running the associated tests before moving on.  
6. **Once all tests pass**, run the final integration test (Prompt 8).  
7. **Build a source distribution** (`python -m build`) and/or **Docker image** (`docker build -t joke-duplicate-detection .`).  
8. **Publish** the package or distribute the Docker image as needed.

> **Tip**: Keep the test suite in `tests/`. Use `pytest`’s `--maxfail=1` to catch failures early.  
> **Tip**: In the real environment, secure the `DB_CONFIG` via environment variables or a `.env` file and load them with `python-dotenv` (optional).

These prompts give a clear, incremental path from scratch to a fully‑tested, packaged duplicate‑search tool that satisfies the specification.