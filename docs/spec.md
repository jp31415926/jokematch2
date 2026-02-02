# 📚 Specification – Joke‑Duplicate‑Detection Suite
**Author:** [Your Name]
**Date:** 2026‑01‑28

> *This spec covers the entire “duplicate‑search” toolchain that will run on a Linux PC with Python 3.11+.  The system is split into two **independent pipelines** – one based on TF‑IDF + cosine similarity, the other on a lightweight sentence‑transformer.  Each pipeline has a *build* script that pre‑computes the necessary vectors and a *search* script that takes a single‑file joke and returns the top 10 most similar jokes from the `archives` table.*

---

## 1.  High‑Level Architecture

```
┌─────────────────────┐        ┌───────────────────────┐
│  TF‑IDF Pipeline    │        │ Transformer Pipeline  │
├─────────────────────┤        ├───────────────────────┤
│  build_tfidf.py     │        │  build_tf.py          │
│  search_tfidf.py    │        │  search_tf.py         │
└─────────────────────┘        └───────────────────────┘
```

*Both pipelines* read from the same MySQL table `archives (id INT, title VARCHAR, funny TEXT)`.

| Component | Purpose | Output Files |
|-----------|---------|--------------|
| `*_build.py` | Pull data, compute feature vectors, and persist to disk | `*_vectorizer.pkl` (TF‑IDF) <br> `*_matrix.npz` (TF‑IDF matrix) <br> `*_ids.pkl`, `*_titles.pkl` (meta) <br> `*_vectors.npy` (Transformer) <br> `*_ids.npy`, `*_titles.pkl` (meta) |
| `*_search.py` | Load persisted data, encode input joke, rank by similarity, print top‑10 | stdout – “rank score id title” table |

The **search scripts** are *stateless* – they only load data from disk, not from the database.  
The **build scripts** are *idempotent*: running them again will overwrite the cached files.

---

## 2.  Environment & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `mysql-connector-python` | ≥ 8.0.28 | DB access |
| `scikit-learn` | ≥ 1.1.1 | TF‑IDF vectorizer, cosine similarity |
| `scipy` | ≥ 1.7.3 | Sparse matrix I/O |
| `joblib` | ≥ 1.2.0 | Persist vectorizer |
| `pickle` | stdlib | Persist meta maps |
| `numpy` | ≥ 1.23.0 | Vector arrays |
| `sentence-transformers` | ≥ 2.2.2 | Lightweight transformer (all-MiniLM-L6-v2) |
| `torch` | ≥ 2.0.0 | Underlying engine for sentence‑transformers |
| `tqdm` | optional | Progress bars (nice‑to‑have) |

> **Installation (one‑time):**  
> ```bash
> pip install mysql-connector-python scikit-learn scipy joblib numpy sentence-transformers torch tqdm
> ```

> **Hardware Note:**  
> The PC has 16×2.2 GHz cores and 64 GB RAM – ample for both pipelines.  
> The transformer model (`all-MiniLM-L6-v2`) is ~240 MB and runs comfortably on CPU.

---

## 3.  Database Access

All scripts will connect using the following config, which will be contained in the `db_config.py` file:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_user',
    'password': 'your_password',
    'database': 'your_database',
}
```

They fetch data with a single query:

```sql
SELECT id, title, funny FROM archives;
```

No indexes are required for the current size (~9 k rows).  
If the table grows dramatically, you can add an index on `funny` (`FULLTEXT`) to speed future inserts, but it is *not* needed for the current pipeline.

---

## 4.  TF‑IDF Pipeline

### 4.1 Build Script – `build_tfidf.py`

```bash
python build_tfidf.py
```

#### Steps

1. Connect to MySQL, fetch all rows into `rows = [(id, title, funny), …]`.
2. Extract lists:
   - `ids = [r[0] for r in rows]`
   - `titles = {r[1] for r in rows}`
   - `texts = [r[2] for r in rows]`
3. **Vectorizer**  
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer(
       stop_words='english',
       lowercase=True,
       ngram_range=(1,2),      # unigrams & bigrams
       min_df=2,               # ignore very rare tokens
       max_features=50000      # keep memory reasonable
   )
   tfidf_matrix = vectorizer.fit_transform(texts)  # sparse CSR matrix
   ```
4. **Persist**  
   ```python
   import joblib, pickle, scipy.sparse
   joblib.dump(vectorizer, 'data/tfidf_vectorizer.pkl')
   scipy.sparse.save_npz('data/tfidf_matrix.npz', tfidf_matrix)
   with open('data/tfidf_ids.pkl', 'wb') as f: pickle.dump(ids, f)
   with open('data/tfidf_titles.pkl', 'wb') as f: pickle.dump(titles, f)
   ```
5. Print “TF‑IDF build finished: <n rows> jokes”.

### 4.2 Search Script – `search_tfidf.py`

```bash
python search_tfidf.py /path/to/joke.txt
```

#### Steps

1. **Load**:
   ```python
   vectorizer = joblib.load('data/tfidf_vectorizer.pkl')
   tfidf_matrix = scipy.sparse.load_npz('data/tfidf_matrix.npz')
   with open('data/tfidf_ids.pkl', 'rb') as f: ids = pickle.load(f)
   with open('data/tfidf_titles.pkl', 'rb') as f: titles = pickle.load(f)
   ```
2. **Read input file** (plain text).  
   ```python
   with open(sys.argv[1], 'r', encoding='utf-8') as f:
       input_text = f.read()
   ```
3. **Transform**:
   ```python
   query_vec = vectorizer.transform([input_text])  # 1×n sparse vector
   ```
4. **Similarity** – cosine similarity via `linear_kernel` (fast on sparse matrices):
   ```python
   from sklearn.metrics.pairwise import linear_kernel
   scores = linear_kernel(query_vec, tfidf_matrix).flatten()
   ```
5. **Rank & output**:
   ```python
   import numpy as np
   top_indices = np.argpartition(scores, -10)[-10:][::-1]  # descending
   print(f"{'Rank':<6} {'Score':<10} {'ID':^5}   Title")
   for rank, idx in enumerate(top_indices, start=1):
       print(f"{rank:>2}     {scores[idx]:6.4f} {ids[idx]:>5d}   {titles[ids[idx]]:<40}")
   ```
6. End of script.

---

## 5.  Transformer Pipeline

> **Model** – `all-MiniLM-L6-v2` (≈ 240 MB).  It is CPU‑friendly and has a 384‑dimensional output vector.

### 5.1 Build Script – `build_tf.py`

```bash
python build_tf.py
```

#### Steps

1. Load data from DB as before (`ids`, `titles`, `texts`).
2. **Encode** all jokes:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   vectors = model.encode(texts, batch_size=64, show_progress_bar=True)
   ```
3. **Persist**:
   ```python
   import numpy as np, pickle
   np.save('data/tf_vectors.npy', vectors)        # shape (n, 384)
   np.save('data/tf_ids.npy', np.array(ids))      # shape (n,)
   with open('data/tf_titles.pkl', 'wb') as f: pickle.dump(titles, f)
   ```
4. Print “Transformer build finished: <n rows> jokes”.

### 5.2 Search Script – `search_tf.py`

```bash
python search_tf.py /path/to/joke.txt
```

#### Steps

1. **Load**:
   ```python
   model = SentenceTransformer('all-MiniLM-L6-v2')
   vectors = np.load('data/tf_vectors.npy')
   ids = np.load('data/tf_ids.npy')
   with open('data/tf_titles.pkl', 'rb') as f: titles = pickle.load(f)
   ```
2. **Read input** file (plain text).
3. **Encode query**:
   ```python
   query_vec = model.encode(input_text, show_progress_bar=False)  # 384‑d
   ```
4. **Similarity** – cosine similarity (dot product / norms):
   ```python
   import numpy as np
   scores = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec))
   ```
5. **Rank & output** (same table format as TF‑IDF):
   ```python
   top_indices = np.argpartition(scores, -10)[-10:][::-1]
   print(f"{'Rank':<6} {'Score':<10} {'ID':^5}   Title")
   for rank, idx in enumerate(top_indices, start=1):
       print(f"{rank:>2}     {scores[idx]:6.4f} {ids[idx]:>5d}   {titles[ids[idx]]:<40}")
   ```

> **Note:** The `SentenceTransformer` loads the model into memory (~ 300 MB) on the first call to `search_tf.py`.  Subsequent calls reuse the same process memory if the script is kept alive (e.g., via a long‑running server), but for a simple CLI you’ll pay that cost each time.

---

## 6.  File Naming & Organization

```
project_root/
├── data/
├── docs/
├── samples/
├── tests/
├── db.py
├── db_config.py
├── tfidf_vectorizer.pkl
├── tfidf_matrix.npz
├── tfidf_ids.pkl
├── tfidf_titles.pkl
├── tf_vectors.npy
├── tf_ids.npy
├── tf_titles.pkl
├── build_tfidf.py
├── search_tfidf.py
├── build_tf.py
└── search_tf.py
```

> Keep all files in the same directory except tests, docs, data and samples.

---

## 7.  Running a Test

1. **Initial build** (once after populating `archives`):

```bash
python build_tfidf.py
python build_tf.py
```

2. **Search a sample joke**:

Three samples are provided: `samples/test-email1.eml`, `samples/test-email1.eml`, `samples/test-email1.eml`. All three should output something and not fail.

```bash
# TF‑IDF
python search_tfidf.py samples/test-email1.eml
python search_tfidf.py samples/test-email2.eml
python search_tfidf.py samples/test-email3.eml

# Transformer
python search_tf.py samples/test-email1.eml
python search_tf.py samples/test-email2.eml
python search_tf.py samples/test-email3.eml
```

Both scripts will output results similar to the following:

```
Rank   Score       ID     Title
 1     0.3463     12628   Why did the chicken cross the road?
 2     0.2251        36   A classic joke about a chicken
 3     0.1615       521   Top Ten Reasons Eve Was Created
...
```
Each column has one space between it and the next column, except between ID and Title, which is 3 spaces. The widths of each column follows:
| Column | Width |
|--------|-------|
| Rank   |   6   |
| Score  |  10   |
| ID     |   5   |
| Title  |  40   |

---

## 8.  Extending the Pipeline

Potential features that might be implemented follow.

| Feature | How to Add |
|---------|------------|
| **Duplicate Test** | Instead of ranking, return a single string indicating if the top score is above a given threshold. Add a CLI flag `--threshold 0.85`. |
| **Database Duplicate Test** | Instead of reading a provided file, use a CLI flag to pass a joke ID that will be compared to all the other jokes in the database. |
| **Adjust similarity threshold** | After ranking, filter `scores >= THRESHOLD`. Add a CLI flag `--threshold 0.85`. |
| **Support batch searching** | Accept a directory and loop over each file. |
| **Persist scores for quick re‑search** | Store a KD‑tree or FAISS index for the transformer vectors. |
| **Add a quick‑look CLI for the email‑parsing step** | Implement `extract_jokes.py` that reads a `.eml` file, strips headers/HTML, and writes each joke to a temporary file that can be fed to the search scripts. |

---

## 9.  Testing & Validation

1. **Unit Tests** – Verify that `build_*` scripts produce non‑empty files and that `search_*` return a score ≥ 0 and ≤ 1 for a known joke.
2. **Accuracy Test** – User will provide a test set of 50 jokes. Run both pipelines, and compare top‑10 lists.  Adjust `min_df`, `ngram_range`, and transformer model if necessary.
3. **Performance Benchmarks** – Measure build time, memory usage, and search latency on the target machine.  Aim for < 5 s for a single search.

---

## 10.  Deployment Checklist

1. **Database Credentials** – Securely store `DB_CONFIG` in a `.env` file or environment variables.
2. **Python Virtual Environment** – Create `venv`, activate, and install dependencies.
3. **Schedule Builds** – Run `build_tfidf.py` and `build_tf.py` nightly or whenever new jokes are inserted.
4. **Use as CLI** – Integrate `search_tfidf.py`/`search_tf.py` into your email workflow (e.g., via a cron job that processes each new `.eml` file).
5. **Monitor** – Log the top‑10 matches and manually review any flagged duplicates to fine‑tune thresholds.

---

## 11.  Summary of Commands

| Purpose | Command |
|---------|---------|
| Build TF‑IDF features | `python build_tfidf.py` |
| Search with TF‑IDF | `python search_tfidf.py /path/to/joke.eml` |
| Build Transformer features | `python build_tf.py` |
| Search with Transformer | `python search_tf.py /path/to/joke.eml` |

---

### Final Note

This spec gives you two self‑contained, production‑ready pipelines.  
Feel free to tweak the vectorizer parameters, model choice, or output formatting to suit your evolving needs. Happy hacking!