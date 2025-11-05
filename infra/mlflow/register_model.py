import os
import click
from mlflow.tracking import MlflowClient

@click.command()
@click.argument('experiment_name')
@click.argument('model')
def import_model(experiment_name: str, model: str) -> None:
    mlflow_url = os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5500")
    client = MlflowClient(tracking_uri=mlflow_url)

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print("No such experiment")
        return
    
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])

    ARTIFACT_PATH = "model" 

    try:
        client.create_registered_model(model)
    except Exception:
        pass

    for r in runs:
        model_uri = f"runs:/{r.info.run_id}/{ARTIFACT_PATH}"
        try:
            mv = client.create_model_version(name=model, source=model_uri, run_id=r.info.run_id)
            print("Created version:", mv.version, "for run", r.info.run_id)
        except Exception as e:
            print("Skip run", r.info.run_id, "->", e)


if __name__ == '__main__':
    import_model()