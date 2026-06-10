fmt:
    uv run ruff check src tests scripts examples --fix-only
    uv run ruff format src tests scripts examples
    uv run mypy src tests scripts examples

check:
    uv run ruff check src tests scripts examples
    uv run ruff format --check src tests scripts examples
    uv run mypy src tests scripts examples