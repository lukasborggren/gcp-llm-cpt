# pylint: disable=import-outside-toplevel, redefined-outer-name, reimported, too-many-locals
"""Kubeflow pipeline components."""

from typing import NamedTuple, Optional

import tomlkit
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
from kfp import dsl

from src.config import gcp

with open("poetry.lock", "r", encoding="utf-8") as f:
    packages = tomlkit.parse(f.read()).get("package", [])
VERSIONED: dict[str, str] = {
    pkg["name"]: f"{pkg['name']}=={pkg['version']}" for pkg in packages
}


@dsl.component(
    base_image=gcp.IMAGES["python"], packages_to_install=[VERSIONED["pyarrow"]]
)
def create_dataset(uri: str, name: Optional[str] = None) -> dsl.Dataset:
    """Post-hoc creation of a KFP Dataset.

    Intended to be used after running a Dataproc serverless Spark job,
    that do not produce any artifacts(s) itself.

    Args:
        uri (str):
            The URI where the dataset is stored.
        name (str, optional):
            The name of the dataset. Defaults to None.

    Returns:
        kfp.dsl.Dataset: The created KFP Dataset object.
    """

    import os

    import pyarrow.parquet as pq

    schema = pq.read_schema(os.path.join(uri, "_metadata"))
    return dsl.Dataset(
        name=name,
        uri=uri,
        metadata={
            "schema": {field.name: str(field.type) for field in schema},
            "num_rows": int(schema.metadata.get(b"tot_num_rows", -1)),
        },
    )


@dsl.component(base_image=gcp.IMAGES["python"])
def set_exact_dedup_args(
    source: dsl.Artifact, normal_form: str
) -> NamedTuple("Outputs", [("args", list), ("destination", str)]):  # type: ignore
    """Set arguments for Spark application and URI for output artifact.

    Args:
        source (kfp.dsl.Artifact):
            BigQuery table artifact.
        normal_form (str):
            The normalization form to be used in the deduplication process.
    Returns:
        NamedTuple:
            A named tuple containing:
                - args (list): A list of arguments for the exact deduplication process.
                - destination (str): The destination URI for the deduplication results.
    """

    uri = f"{dsl.get_uri().split('/set-spark-args')[0]}/exact-deduplication"
    table_name = ".".join(
        map(source.metadata.get, ["projectId", "datasetId", "tableId"])
    )
    return [
        "exact-deduplication",
        "--source",
        table_name,
        "--normal-form",
        normal_form,
        "--destination",
        uri,
    ], uri


@dsl.component(base_image=gcp.IMAGES["python"])
def set_fuzzy_dedup_args(
    source: dsl.Artifact, ngram_len: int, minhash_perm: int, sim_threshold: float
) -> NamedTuple("Outputs", [("args", list), ("destination", str)]):  # type: ignore
    """Set arguments for Spark application and URI for output artifact.

    Args:
        source (kfp.dsl.Artifact):
            Source dataset artifact containing the data to be deduplicated.
        ngram_len (int):
            The length of n-grams to be used in the deduplication process.
        minhash_perm (int):
            The number of minhash permutations to be used.
        sim_threshold (float):
            The similarity threshold for deduplication.

    Returns:
        NamedTuple:
            A named tuple containing:
                - args (list): A list of arguments for the fuzzy deduplication process.
                - destination (str): The URI where the deduplicated data will be stored.
    """

    uri = f"{dsl.get_uri().split('/set-spark-args')[0]}/fuzzy-deduplication"
    return [
        "fuzzy-deduplication",
        "--source",
        source.uri,
        "--ngram-length",
        str(ngram_len),
        "--minhash-permutations",
        str(minhash_perm),
        "--similarity-threshold",
        str(sim_threshold),
        "--destination",
        uri,
    ], uri


@dsl.component(base_image=gcp.IMAGES["python"])
def parse_tokenizer_config(model: dsl.Model, train_config: dict) -> dict:
    """Used to ensure correct caching behavior.

    Extracts arguments from the training config that are related to tokenization,
    so that tokenization is not re-triggered for unrelated config changes.

    Args:
        model (kfp.dsl.Model):
            The model object containing the URI to be used.
        train_config (dict):
            The training configuration dictionary containing tokenizer settings.

    Returns:
        dict:
            A dictionary containing the updated tokenizer configuration.
    """
    import os

    train_config["tokenizer"]["path"] = os.path.join(
        model.uri, train_config["tokenizer"]["path"]
    )

    return {"tokenizer": train_config["tokenizer"]}


@dsl.component(base_image=gcp.IMAGES["python"])
def set_tokenization_args(
    source: dsl.Artifact, tokenizer_config: dict
) -> NamedTuple("Outputs", [("args", list), ("destination", str)]):  # type: ignore
    """Set arguments for Spark application and URI for output artifact.

    Args:
        source (kfp.dsl.Artifact):
            The source artifact containing the data to be tokenized.
        tokenizer_config (dict):
            A dictionary containing the tokenizer configuration.

    Returns:
        NamedTuple:
            A named tuple containing:
                - args (list) A list of command-line arguments for the tokenization process.
                - destination (str): The destination URI where the tokenized data will be stored.
    """

    import json

    uri = f"{dsl.get_uri().split('/set-spark-args')[0]}/tokenization"
    return [
        "tokenization",
        "--source",
        source.uri,
        "--tokenizer-config",
        json.dumps(tokenizer_config),
        "--destination",
        uri,
    ], uri


@dsl.component(
    base_image=gcp.IMAGES["python"],
    packages_to_install=[
        VERSIONED["huggingface-hub"],
        VERSIONED["google-cloud-storage"],
        VERSIONED["google-cloud-secret-manager"],
    ],
)
def download_hf_repo(  # pylint: disable=dangerous-default-value
    repo_id: str,
    bucket_name: str,
    hf_secret_name: str,
    gcs_model_dir: str,
    ignore_patterns: list = ["original/consolidated.00.pth"],
) -> dsl.Model:
    """Download Hugging Face repository and upload it to GCS.

    Args:
        repo_id (str):
            The ID of the Hugging Face repository to download.
        bucket_name (str):
            The name of the GCS bucket to upload to.
        hf_secret_name (str):
            The name of the secret in Google Cloud Secret Manager containing the Hugging Face token.
        gcs_model_dir (str):
            The directory in the GCS bucket where the model should be stored.
        ignore_patterns (list, optional):
            List of file patterns to ignore during download.

    Returns:
        ksp.dsl.Model:
            A KFP model artifact pointing to the GCS model directory.
    """

    import os
    import tempfile
    from pathlib import Path

    from google.cloud import secretmanager
    from google.cloud.storage import Client, transfer_manager
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    token = (
        secretmanager.SecretManagerServiceClient()
        .access_secret_version(request={"name": hf_secret_name})
        .payload.data.decode("UTF-8")
    )
    bucket = Client().bucket(bucket_name)

    with tempfile.TemporaryDirectory() as tmpdirname:
        snapshot_download(
            repo_id, local_dir=tmpdirname, token=token, ignore_patterns=ignore_patterns
        )

        results = transfer_manager.upload_many_from_filenames(
            bucket,
            [
                str(f.relative_to(Path(tmpdirname)))
                for f in Path(tmpdirname).rglob("*")
                if ".cache" not in f.parts and not os.path.isdir(f)
            ],
            source_directory=tmpdirname,
            blob_name_prefix=os.path.join(gcs_model_dir, repo_id) + "/",
        )

    for result in results:
        if isinstance(result, Exception):
            raise result

    return dsl.Model(uri=os.path.join("gs://", bucket_name, gcs_model_dir, repo_id))


@dsl.component(base_image=gcp.IMAGES["cuda"])
def train(
    pretrained_model: dsl.Model, train_data: dsl.Dataset, recipe_uri: str, config: dict
) -> dsl.Model:
    """Continued pre-training of an LLM using torchtune and torchrun.

    Currently only supports single-node multi-worker training.

    Args:
        pretrained_model (kfp.dsl.Model):
            The pre-trained model to fine-tune.
        train_data (kfp.dsl.Dataset):
            The dataset to use for training.
        recipe_uri (str):
            The URI of the training recipe.
        config (dict):
            Configuration dictionary for training parameters.

    Returns:
        kfp.dsl.Model:
            The trained model.
    """

    import json
    import os
    from urllib.parse import urlparse

    import torch
    from google.cloud import aiplatform
    from google.cloud.storage import Client, transfer_manager
    from kfp import dsl
    from torch.distributed.run import get_args_parser, run
    from torchtune import utils

    log = utils.get_logger("DEBUG")
    model = dsl.Model(uri=dsl.get_uri())

    log.info("Downloading training recipe from %s", recipe_uri)
    uri = urlparse(recipe_uri)
    # my-bucket, dir/.../file.py <- gs://my-bucket/dir/.../file.py
    bucket_name, filepath = uri.netloc, uri.path.lstrip("/")
    filename = os.path.basename(filepath)  # file.py
    bucket = Client().bucket(bucket_name)
    bucket.blob(filepath).download_to_filename(filename)

    # It would likely be more efficient to use Dataflux to download the
    # the checkpoint files directly from GCS when the model is loaded.
    # However, it becomes very tedious to implement if Safetensors are used,
    # rather than cases where `torch.load` is used.
    log.info("Downloading pre-trained model from %s", pretrained_model.uri)
    # dir/.../subdir <- gs://my-bucket/dir/.../subdir
    model_dir = urlparse(pretrained_model.uri).path.lstrip("/")
    transfer_manager.download_many_to_path(
        bucket, [blob.name for blob in bucket.list_blobs(prefix=f"{model_dir}/")]
    )

    # Overriding arguments from the YAML config.
    config["bucket_name"] = bucket_name
    config["output_dir"] = urlparse(model.uri).path.lstrip("/")
    config["tokenizer"]["path"] = os.path.join(model_dir, config["tokenizer"]["path"])
    config["checkpointer"]["checkpoint_dir"] = model_dir
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["dataset"]["uri"] = train_data.uri
    os.makedirs(config["output_dir"], exist_ok=True)

    # HACK: This is inspired by the entrypoint of the torchtune CLI.
    # The argparser from the torchrun CLI is used to get defaults for optional arguments.
    parser = get_args_parser()
    args = parser.parse_args([filename, "--config", json.dumps(config)])
    args.standalone = True
    args.nproc_per_node = "auto"

    log.info("Launching torchrun")

    aiplatform.start_upload_tb_log(
        tensorboard_experiment_name=os.environ["CLOUD_ML_JOB_ID"],
        logdir=config["metric_logger"]["log_dir"],
        tensorboard_id=os.environ["AIP_TENSORBOARD_RESOURCE_NAME"].rsplit("/", 1)[1],
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["CLOUD_ML_REGION"],
    )
    try:
        run(args)
    finally:
        aiplatform.end_upload_tb_log()

    return model


# TODO: the worker pool spec should be set dynamically
custom_training_job = create_custom_training_job_from_component(
    train,
    replica_count=1,
    machine_type="g2-standard-48",
    accelerator_type="NVIDIA_L4",
    accelerator_count=4,
    service_account=gcp.SERVICE_ACCOUNT,
    tensorboard=gcp.TENSORBOARD,
    base_output_directory=f"{gcp.BUCKET}/custom-job-outputs",
    enable_web_access=True,
)
