import yaml
import os
import json

import mlflow
from mlflow.tracking import MlflowClient

with open("./src/config/env_config.json", 'r') as f:
    mlflow_config = json.load(f)["mlflow"]
    for key,value in mlflow_config.items():
        os.environ[key] = value

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow_client = MlflowClient()


