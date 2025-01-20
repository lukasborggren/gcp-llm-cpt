# Continued Pre-Training of Large Language Models on Google Cloud Platform
_Lukas Borggren, Linköping University and Bonnier News_

This project showcases how to perform distributed continued pre-training of large language models (LLMs) on [Google Cloud Platform (GCP)](https://cloud.google.com/?hl=sv).


Continued (or continual) pre-training is a method for specializing a foundational LLM, for example, to a new domain, language or task. By continuing to train a general-purpose LLM on new data, the model can be adapted using relatively small amounts of data and compute. This repository contains an end-to-end solution for creating a dataset and subsequently perform distributed LLM pre-training on it. It is built as a [Kubeflow pipeline](https://www.kubeflow.org/docs/components/pipelines/) running on [Vertex AI](https://cloud.google.com/vertex-ai) and [Dataproc](https://cloud.google.com/dataproc), utilizing tools like [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) and [torchtune](https://pytorch.org/torchtune/stable/index.html), and the aim is to handle arbitrarily large (within reason) data and model sizes. Currently, parallelism through [fully sharded data parallelism (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) is supported, but not alternatives such as [DeepSpeed](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/). To the best of the author's knowledge, this is the only publicly available example of end-to-end LLM pre-training on Vertex AI Pipelines.

[Presentation](https://docs.google.com/presentation/d/1efpcJwxcghhQlm-Mp9fsz6X3IbD76yIZ2NswFY0HhO8) of the project. In the presentation, the pipeline is demonstrated by training on Swedish news articles. However, since the data is proprietary, no information about it is included in the repository.

## Setup
### Developement Environment
This project uses [pyenv](https://github.com/pyenv/pyenv) for Python version management and [Poetry](https://github.com/python-poetry/poetry) for dependency management. The [Google Cloud CLI](https://cloud.google.com/cli) is used for managing cloud resources and services. To set up a local development environment, run the following commands from the repository root:

```
pyenv local 3.11.9
poetry env use $(pyenv which python)
poetry install
gcloud auth login --update-adc
```

### Google Cloud Project
Create a [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and enable the following services:
* [Cloud Build](https://cloud.google.com/build) – for building Docker container images
* [Artifact Registry](https://cloud.google.com/artifact-registry) – for storing Docker container images
* [BigQuery](https://cloud.google.com/bigquery) – for storing structured data
* [Cloud Storage](https://cloud.google.com/storage) – for storing unstructured data
* [Secret Manager](https://cloud.google.com/secret-manager) – for storing sensitive data
* [Compute Engine](https://cloud.google.com/compute) – for running virtual machines (VMs)
* [Dataproc](https://cloud.google.com/dataproc) – for running Spark workloads
* [Vertex AI](https://cloud.google.com/vertex-ai) – for orchestrating and tracking training


Also, create a service account – or use the [Compute Engine default service account](https://cloud.google.com/compute/docs/access/service-accounts#default_service_account) – and give it the predefined Identity and Access Management (IAM) roles:
* BigQuery Job User 
* Dataproc Worker
* Vertex AI User
* Secret Manager Secret Accessor

To utilize Dataproc Serverless for Spark, it is required to [configure a Virtual Private Cloud subnetwork](https://cloud.google.com/dataproc-serverless/docs/concepts/network). For storing resources in the cloud, create a [Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) and an [Artifact Registry Docker repository](https://cloud.google.com/artifact-registry/docs/repositories/create-repos). If [gated Hugging Face models](https://huggingface.co/docs/hub/models-gated) are used, it is also required to create a [User Access Token](https://huggingface.co/docs/hub/security-tokens) and upload it to [Secret Manager](https://cloud.google.com/secret-manager/docs/creating-and-accessing-secrets). To track training progress in real-time, set up a [Vertex AI TensorBoard instance](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-setup). Also, the source data is expected to be stored in a [BigQuery table](https://cloud.google.com/bigquery/docs/tables) named `<PROJECT_ID>.default.texts` with columns `id` and `text`.

### Configuration
This project utilizes three main configuration levels. Firstly, the [GCP config](src/config/gcp.py) holds all information about the cloud resources and should be updated based on the previous step. Secondly, the [training config(s)](src/config/llama3_2_1b.yaml) are YAML files for controlling the custom [torchtune](https://github.com/pytorch/torchtune) training recipe in this repository. Lastly the [pipeline job config](src/config/job.py) holds all arguments for the pipeline execution.

## Run
Before running the pipeline, two Docker images must be built and pushed to the Docker repository. Using Cloud Build, this can all be performed in the cloud by running:

```
bash images/build.sh cuda
bash images/build.sh spark
```

Then, to execute a pipeline, simply run `poetry run python src/main.py`. This will compile and upload the Kubeflow pipeline to GCS, along with scripts for preprocessing and training. Thereafter, a pipeline job will be submitted to Vertex AI and the progress can be tracked in the [GCP console](https://console.cloud.google.com/vertex-ai/pipelines/runs).

## Acknowledgements
This project was partially supported by the Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to fufill the requirements to pass the WASP Graduate School Course Scalable Data Science and Distributed Machine Learning - ScaDaMaLe-WASP-UU-2024 at https://lamastex.github.io/ScaDaMaLe. Computing infrastructure for learning was supported by Databricks Inc.'s Community Edition. The course was Industrially sponsored by Jim Dowling of Logical Clocks AB, Stockholm, Sweden, Reza Zadeh of Matroid Inc., Palo Alto, California, USA, and Andreas Hellander & Salman Toor of Scaleout Systems AB, Uppsala, Sweden.