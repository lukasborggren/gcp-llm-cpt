"""Configuration for Google Cloud Platform (GCP) and related resources."""

import subprocess

kfp_param = bool | int | float | str | list | dict

PYTHON_VERSION: str = subprocess.check_output(["pyenv", "local"]).decode().strip()
CUDA_VERSION: str = "12.4.1"
SPARK_RUNTIME_VERSION: str = "2.2"

PROJECT_ID: str = "<REPLACE>"
PROJECT_NUMBER: str = "<REPLACE>"
REGION: str = "<REPLACE>"

REPOSITORY: str = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/default"
SERVICE_ACCOUNT: str = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
SUBNETWORK: str = f"projects/{PROJECT_ID}/regions/{REGION}/subnetworks/default"
HF_SECRET_NAME: str = f"projects/{PROJECT_ID}/secrets/huggingface-token/versions/1"
TENSORBOARD: str = f"projects/{PROJECT_ID}/locations/{REGION}/tensorboards/<REPLACE>"

BUCKET_NAME: str = f"vertex-ai-pipelines-staging-{REGION}-{PROJECT_NUMBER}"
BUCKET: str = f"gs://{BUCKET_NAME}"
STAGING_PATH: str = f"{BUCKET}/kubeflow-artifacts"
TEMPLATE_DIR: str = "kubeflow-pipelines"
APPLICATION_DIR: str = "spark-applications"
RECIPE_DIR: str = "torchtune-recipes"
MODEL_DIR: str = "huggingface-models"

IMAGES: dict[str, str] = {
    "python": f"python:{PYTHON_VERSION}-slim",
    "cuda": f"{REPOSITORY}/cuda",
    "spark": f"{REPOSITORY}/spark",
}
