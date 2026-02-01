#!/usr/bin/bash

rm -rf __pycache__ .pytest_cache
rm -rf tests/__pycache__
rm tfidf_ids.pkl tfidf_matrix.npz tfidf_titles.pkl tfidf_vectorizer.pkl
