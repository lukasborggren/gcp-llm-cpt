"""Configuration for a pipeline job."""

from typing import cast

import torchtune.config
from omegaconf import OmegaConf

from src.config import gcp

PIPELINE_NAME: str = "pretraining"
PIPELINE_FILE: str = f"{PIPELINE_NAME}.yaml"
PREPROCESS_FILE: str = "src/scripts/preprocessing.py"
TRAIN_RECIPE_FILE: str = "src/scripts/train_recipe.py"
TRAIN_CONFIG_FILE: str = "src/config/llama3_2_1b.yaml"

# This query is very simplified to avoid exposing any sensitive information.
# It mostly demonstrates which columns are required to run the pipeline.
QUERY: str = f"""
    SELECT
        id,
        text
    FROM
        `{gcp.PROJECT_ID}.default.texts`
"""

train_config = OmegaConf.load(TRAIN_CONFIG_FILE)
torchtune.config.validate(train_config)

ARGS: dict[str, gcp.kfp_param] = {
    "pretrained_hf_model": "meta-llama/Llama-3.2-1B",
    "source_query": QUERY,
    "query_job_config": {
        "destinationTable": {
            "projectId": gcp.PROJECT_ID,
            "datasetId": "default",
            "tableId": "articles",
        },
        "writeDisposition": "WRITE_TRUNCATE",
    },
    "normal_form": "NFKD",
    "ngram_len": 5,
    "minhash_perm": 2,
    "sim_threshold": 0.2,
    "preprocess_file_uri": f"{gcp.BUCKET}/{gcp.APPLICATION_DIR}/{PREPROCESS_FILE}",
    "train_recipe_file_uri": f"{gcp.BUCKET}/{gcp.RECIPE_DIR}/{TRAIN_RECIPE_FILE}",
    "train_config": cast(dict, OmegaConf.to_container(train_config)),
}

# Compute resources for the training step of the pipeline.
# Other options are available at: https://cloud.google.com/vertex-ai/docs/training/configure-compute
WORKER_POOL_SPEC: dict[str, str | int] = {
    "machine_type": "g2-standard-48",
    "accelerator_type": "NVIDIA_L4",
    "accelerator_count": 4,
    "replica_count": 1,
}
