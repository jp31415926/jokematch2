# tests/test_db.py
"""
pytest test suite for the lightweight MySQL connector module.

Run with:  pytest -q
"""

from __future__ import annotations

import pytest
from typing import List, Tuple

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:
    mysql = None

# Import the function under test
from db import fetch_jokes, DBConnectionError


@pytest.fixture(scope="module")
def db_configured() -> bool:
    """
    Checks whether a live MySQL database is available using the
    credentials defined in ``db.DB_CONFIG``.
    """
    if mysql is None:
        return False
    try:
        conn = mysql.connector.connect(**fetch_jokes.__globals__["DB_CONFIG"])
        conn.close()
        return True
    except MySQLError:
        return False


@pytest.mark.skipif(not db_configured(), reason="MySQL database not configured.")
def test_fetch_jokes_returns_list_of_tuples() -> None:
    result = fetch_jokes()
    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(r, tuple) for r in result), "All items must be tuples"
    # Check that tuples have the expected structure.
    for r in result:
        assert len(r) == 3, "Tuple must have 3 elements"
        assert isinstance(r[0], int), "ID should be an int"
        assert isinstance(r[1], str), "Title should be a str"
        assert isinstance(r[2], str), "Funny field should be a str"


@pytest.mark.skipif(not db_configured(), reason="MySQL database not configured.")
def test_fetch_jokes_non_empty() -> None:
    """
    Ensure that the ``archives`` table contains at least one record.
    """
    result = fetch_jokes()
    assert len(result) > 0, "The archives table should contain at least one joke"