install-kernel:
	uv run python -m ipykernel install --user --name=qonto-ml-tech-assignment --display-name="Qonto ML Tech Assignment"

mypy:
	uv run mypy api/ tests/

black:
	black --check api/ tests/

black-fix:
	black api/ tests/

ruff:
	ruff check api/ tests/ --fix

tests-docker:
	docker-compose run test

test:
	uv run pytest tests/ --cov=api --cov-report=html