# Joke Duplicate Detection Suite - AGENTS.md
## Project Overview
This is a joke duplicate detection suite with two pipelines: TF-IDF and sentence-transformer based. The system is designed to detect duplicate jokes using similarity search.

# Build/Lint/Test Commands

# Setup
```
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
```
## Install dependencies
```
pip install mysql-connector-python scikit-learn scipy joblib numpy sentence-transformers torch tqdm
```
# Running Tests
```
## Run all tests
pytest
```
## Run specific test file
```
pytest tests/test_db.py
```
## Run with verbose output
```
pytest -v
```

## Run with coverage
```
pytest --cov=src
```

## Run a single test function
```
pytest tests/test_db.py::test_fetch_jokes_returns_list_of_tuples
```

# Linting
```
## Lint with flake8
flake8 src/
```

## Lint with pylint
```
pylint src/
```
## Type checking with mypy
```
mypy src/
```

# Code Style Guidelines
## Imports
- All imports should be at the top of the file
- Use absolute imports when possible
- Group imports in order: standard library, 3rd party, local
- Use `from `__future__` import annotations` for forward references
- Import type hints with `from typing import` for clarity
## Formatting
- Use 2 spaces for indentation (no tabs)
- Maximum line length of 88 characters (PEP8)
- Use PEP 8 style naming conventions
- Add blank lines between functions and classes at module level
- Use blank lines to separate logical sections within functions
## Types and Type Hints
- Always add type hints for function parameters and return values
- Use `List[T]`, `Dict[K, V]` etc. from typing module
- Use typing.Protocol for structural type checking
- Prefer `typing.TYPE_CHECKING` for imports only used for type hints
## Naming Conventions
- Use `snake_case` for functions, variables, and attributes
- Use `PascalCase` for classes
- Use `UPPER_CASE` for constants
- Use `camelCase` for private attributes (when required by library)
## Error Handling
- Create custom exceptions when appropriate
- Log errors using the standard logging module
- Handle database connections gracefully
- Provide informative error messages to users
- Use try-except blocks with specific exception types
## Documentation
- Write docstrings for all functions and classes (Google style)
- Include parameter descriptions and return values
- Use type hints in docstrings where necessary
- Document all public methods in the module
## Best Practices
- Keep functions small and focused on single responsibility
- Write idempotent functions where possible
- Use context managers (`with` statements) for resource management
- Handle connection errors gracefully
- Include `__main__` guards for executable modules
- Use logging instead of print statements
- Write unit tests for all functions
- Test database connectivity properly
- Validate inputs and handle edge cases

This information was gathered from analyzing the project structure, code files, and documentation. The project follows Python best practices and includes proper test automation with pytest.