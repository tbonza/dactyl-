
format:
	poetry run ruff format

lint:
	poetry run ruff check

test:
	poetry run pytest tests
