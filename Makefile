install-kernel:
	uv run python -m ipykernel install --user --name=ml-tech-assignment --display-name="ML Tech Assignment"

set-kernel-in-notebooks:
	python kernel.py

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

up:
	docker-compose up api

ploomber:
	make install-kernel && make set-kernel-in-notebooks && ploomber build

jupyter-up:
	docker-compose up jupyter

lint:
	make mypy & make black & make ruff

test:
	uv run pytest tests/ --cov=api --cov-report=html

venv_on:
	source .venv/bin/activate

venv_off:
	source .venv/bin/deactivate