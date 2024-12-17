"""Script for compiling and running a Kubeflow pipeline on Vertex AI."""

import tempfile

from google.cloud import aiplatform, storage
from kfp import compiler

from src import pipelines
from src.config import gcp, job

gcs = storage.Client(project=gcp.PROJECT_ID)
blob = gcs.bucket(gcp.BUCKET_NAME).blob(f"{gcp.TEMPLATE_DIR}/{job.PIPELINE_FILE}")

with tempfile.TemporaryDirectory() as tmpdirname:
    local_path = f"{tmpdirname}/{job.PIPELINE_FILE}"
    compiler.Compiler().compile(
        pipeline_func=pipelines.pretraining, package_path=local_path
    )
    blob.upload_from_filename(local_path)

# These files have to be individually uploaded to GCS
# (rather than being part of a Docker container) because
# of how their correpsonding jobs are launched.
gcs.bucket(gcp.BUCKET_NAME).blob(
    f"{gcp.APPLICATION_DIR}/{job.PREPROCESS_FILE}"
).upload_from_filename(job.PREPROCESS_FILE)

gcs.bucket(gcp.BUCKET_NAME).blob(
    f"{gcp.RECIPE_DIR}/{job.TRAIN_RECIPE_FILE}"
).upload_from_filename(job.TRAIN_RECIPE_FILE)

aiplatform.init(
    project=gcp.PROJECT_ID, location=gcp.REGION, staging_bucket=gcp.STAGING_PATH
)
pipeline_job = aiplatform.PipelineJob(
    display_name=job.PIPELINE_NAME,
    template_path=f"{gcp.BUCKET}/{blob.name}",
    parameter_values=job.ARGS,
    enable_caching=True,
)

pipeline_job.submit(service_account=gcp.SERVICE_ACCOUNT)
