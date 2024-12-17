# pylint: disable=no-member,not-callable,no-value-for-parameter,too-many-arguments,too-many-positional-arguments
"""Kubeflow pipelines for dataset creation and (continued) LLM pre-training."""

from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.dataproc import DataprocPySparkBatchOp
from kfp import dsl

from src.config import gcp
from components import (
    create_dataset,
    custom_training_job,
    download_hf_repo,
    parse_tokenizer_config,
    set_exact_dedup_args,
    set_fuzzy_dedup_args,
    set_tokenization_args,
)


@dsl.pipeline
def exact_deduplication(
    main_python_file_uri: str, source: dsl.Artifact, normal_form: str
) -> dsl.Dataset:
    """Wrapper around `DataprocPySparkBatchOp` to control caching and artifacts."""

    args_op = (
        set_exact_dedup_args(source=source, normal_form=normal_form)
        .set_caching_options(False)
        .set_display_name("Set Spark Args")
    )

    spark_op = DataprocPySparkBatchOp(
        main_python_file_uri=main_python_file_uri,
        location=gcp.REGION,
        container_image=gcp.IMAGES["spark"],
        runtime_config_version=gcp.SPARK_RUNTIME_VERSION,
        service_account=gcp.SERVICE_ACCOUNT,
        subnetwork_uri=gcp.SUBNETWORK,
        args=args_op.outputs["args"],
    ).set_display_name("Run Spark Workload")

    return (
        create_dataset(uri=args_op.outputs["destination"])
        .after(spark_op)
        .set_display_name("Create Dataset")
        .output
    )


@dsl.pipeline
def fuzzy_deduplication(
    main_python_file_uri: str,
    source: dsl.Artifact,
    ngram_len: int,
    minhash_perm: int,
    sim_threshold: float,
) -> dsl.Dataset:
    """Wrapper around `DataprocPySparkBatchOp` to control caching and artifacts."""

    args_op = (
        set_fuzzy_dedup_args(
            source=source,
            ngram_len=ngram_len,
            minhash_perm=minhash_perm,
            sim_threshold=sim_threshold,
        )
        .set_caching_options(False)
        .set_display_name("Set Spark Args")
    )

    spark_op = DataprocPySparkBatchOp(
        main_python_file_uri=main_python_file_uri,
        location=gcp.REGION,
        container_image=gcp.IMAGES["spark"],
        runtime_config_version=gcp.SPARK_RUNTIME_VERSION,
        service_account=gcp.SERVICE_ACCOUNT,
        subnetwork_uri=gcp.SUBNETWORK,
        args=args_op.outputs["args"],
    ).set_display_name("Run Spark Workload")

    return (
        create_dataset(uri=args_op.outputs["destination"])
        .after(spark_op)
        .set_display_name("Create Dataset")
        .output
    )


@dsl.pipeline
def tokenization(
    main_python_file_uri: str,
    source: dsl.Artifact,
    model: dsl.Model,
    train_config: dict,
) -> dsl.Dataset:
    """Wrapper around `DataprocPySparkBatchOp` to control caching and artifacts."""
    tokenizer_op = parse_tokenizer_config(
        model=model, train_config=train_config
    ).set_display_name("Parse Tokenizer Config")

    args_op = (
        set_tokenization_args(source=source, tokenizer_config=tokenizer_op.output)
        .set_caching_options(False)
        .set_display_name("Set Spark Args")
    )

    spark_op = DataprocPySparkBatchOp(
        main_python_file_uri=main_python_file_uri,
        location=gcp.REGION,
        container_image=gcp.IMAGES["spark"],
        runtime_config_version=gcp.SPARK_RUNTIME_VERSION,
        service_account=gcp.SERVICE_ACCOUNT,
        subnetwork_uri=gcp.SUBNETWORK,
        args=args_op.outputs["args"],
    ).set_display_name("Run Spark Workload")

    return (
        create_dataset(uri=args_op.outputs["destination"])
        .after(spark_op)
        .set_display_name("Create Dataset")
        .output
    )


@dsl.pipeline
def pretraining(
    pretrained_hf_model: str,
    source_query: str,
    query_job_config: dict,
    normal_form: str,
    ngram_len: int,
    minhash_perm: int,
    sim_threshold: float,
    preprocess_file_uri: str,
    train_recipe_file_uri: str,
    train_config: dict,
) -> None:
    """Pipeline for creating a pre-traing dataset and continue pre-training an LLM.

    Args:
        pretrained_hf_model (str):
            The repository ID for the pre-trained Hugging Face model.
        source_query (str):
            The SQL query to retrieve source data.
        query_job_config (dict):
            Configuration for the BigQuery job.
        normal_form (str):
            The normal form to be used in exact deduplication.
        ngram_len (int):
            The length of n-grams for fuzzy deduplication.
        minhash_perm (int):
            The number of permutations for MinHash in fuzzy deduplication.
        sim_threshold (float):
            The similarity threshold for fuzzy deduplication.
        preprocess_file_uri (str):
            The URI of the preprocessing file.
        train_recipe_file_uri (str):
            The URI of the training recipe file.
        train_config (dict):
            Configuration for the training job.
    """

    model_op = download_hf_repo(
        repo_id=pretrained_hf_model,
        bucket_name=gcp.BUCKET_NAME,
        hf_secret_name=gcp.HF_SECRET_NAME,
        gcs_model_dir=gcp.MODEL_DIR,
    ).set_display_name("Download Model")

    query_op = BigqueryQueryJobOp(
        location="EU",
        query=source_query,
        job_configuration_query=query_job_config,
    ).set_display_name("Raw Data Aggregation")

    exact_dedup_op = exact_deduplication(
        main_python_file_uri=preprocess_file_uri,
        source=query_op.outputs["destination_table"],
        normal_form=normal_form,
    ).set_display_name("Exact Deduplication")

    fuzzy_dedup_op = fuzzy_deduplication(
        main_python_file_uri=preprocess_file_uri,
        source=exact_dedup_op.output,
        ngram_len=ngram_len,
        minhash_perm=minhash_perm,
        sim_threshold=sim_threshold,
    ).set_display_name("Fuzzy Deduplication")

    tokenization_op = tokenization(
        main_python_file_uri=preprocess_file_uri,
        source=fuzzy_dedup_op.output,
        model=model_op.output,
        train_config=train_config,
    ).set_display_name("Tokenization")

    _ = custom_training_job(
        location=gcp.REGION,
        pretrained_model=model_op.output,
        train_data=tokenization_op.output,
        recipe_uri=train_recipe_file_uri,
        config=train_config,
    ).set_display_name("Train")
