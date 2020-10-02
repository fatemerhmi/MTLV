import os
from datetime import datetime
import mlflow
from mlflow import log_metric, log_param, log_artifacts

def setup_mlflow(experiment_name, tracking_uri, run_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name, nested=True)

def store_param(key, value):
    log_param(key, value)

def store_metric(key, value, step = None):
    log_metric(key, value, step)

def get_metrics(key):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(mlflow.active_run().info.run_id).data 
    return data.metrics[key]

def get_params(key):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(mlflow.active_run().info.run_id).data 
    return data.params[key]

def finish_mlflowrun():
    mlflow.end_run()

def store_artifact(txt):
    #1. make artifacts directory in mlrun
    ARTIFACT_ROOT = "mlruns/artifacts"
    if not os.path.exists(ARTIFACT_ROOT):
        os.makedirs(ARTIFACT_ROOT)
    
    #2. store the report as .txt file in it
    now = datetime.now()
    FILE_NAME = str(now)+'.txt'
    file = open(ARTIFACT_ROOT + '/' + FILE_NAME, 'w')
    file.write(txt)
    file.close()

    #3. store the artifact in mlflow
    mlflow.log_artifact(ARTIFACT_ROOT + '/' + FILE_NAME)
    
    

