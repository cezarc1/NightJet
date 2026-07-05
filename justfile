set dotenv-load := true

setup:
    uv python install 3.12.12
    uv sync --locked

fmt:
    uv run ruff format .

lint:
    uv run ruff check .

typecheck:
    uv run ty check

test:
    uv run pytest

check: lint typecheck test
