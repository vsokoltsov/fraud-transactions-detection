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

mlflow-up:
	docker-compose up mlflow

mlflow-import-experiment:
	MLFLOW_TRACKING_URI="file:./data/mlflow" export-experiment --experiment $(experiment) --output-dir ./data/mlflow_exported/$(experiment)
	MLFLOW_TRACKING_URI="http://localhost:5500" import-experiment --experiment-name $(experiment) --input-dir ./data/mlflow_exported/$(experiment)

mlflow-export-models-all:
	MLFLOW_TRACKING_URI="file:./data/mlflow" export-models --models 'all' --output-dir ./data/mlflow_exported/models_all

mlflow-import-model:
	MLFLOW_TRACKING_URI="http://localhost:5500" python ./infra/mlflow/register_model.py $(experiment) $(model)
