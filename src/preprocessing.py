"""Spark application for text preprocessing."""

import argparse
import os
import re
import sys
import unicodedata
from typing import Literal
from urllib.parse import urlparse

import nltk
import pyarrow.parquet as pq
import torchtune.config
from google.cloud.storage import Client
from graphframes import GraphFrame
from nltk.util import ngrams
from omegaconf import OmegaConf
from pyspark.broadcast import Broadcast
from pyspark.ml.feature import CountVectorizer, MinHashLSH
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from torchtune.data._utils import truncate
from torchtune.modules.tokenizers import ModelTokenizer

spark = SparkSession.builder.config(
    "spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.13"
).getOrCreate()
# Must be set before calling `tokenize`
tokenizer: Broadcast[ModelTokenizer]


@F.udf(returnType=StringType())
def simplify(text: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"]) -> str:
    """Simplify the input text.

    Do so by normalizing it to the specified Unicode form,
    converting it to lowercase, removing diacritical marks, and replacing
    multiple whitespace characters (including non-breaking spaces) with a single space.

    Args:
        text (str):
            The input text to be simplified.
        form (Literal["NFC", "NFD", "NFKC", "NFKD"]):
            The Unicode normalization form to use.

    Returns:
        str:
            The simplified text.
    """

    return re.sub(
        r"(\s|&nbsp;)+",
        " ",
        "".join(
            c
            for c in unicodedata.normalize(form, text.lower())
            if unicodedata.category(c) != "Mn"
        ),
    ).strip()


@F.udf(returnType=ArrayType(StringType()))
def shingle(text: str, n: int = 5, language="swedish") -> list[str]:
    """Generate n-grams (shingles) from the input text.

    Args:
        text (str):
            The input text to be tokenized and converted into n-grams.
        n (int, optional):
            The number of words in each n-gram. Defaults to 5.
        language (str, optional):
            The language used for tokenization. Defaults to "swedish".

    Returns:
        list[str]:
            A list of n-grams, where each n-gram is a string of `n` words.
    """

    return [
        " ".join(x)
        for x in ngrams(nltk.tokenize.word_tokenize(text, language=language), n)
    ]


@F.udf(returnType=BooleanType())
def non_zero(vector: SparseVector) -> bool:
    """Check if a sparse vector has any non-zero elements.

    Args:
        vector (pyspark.ml.linalg.SparseVector):
            The sparse vector to check.

    Returns:
        bool:
            True if the vector has one or more non-zero elements, False otherwise.
    """

    return vector.numNonzeros() > 0


@F.udf(returnType=StructType([StructField("tokens", ArrayType(IntegerType()))]))
def tokenize(text: str, add_eos: bool = False) -> dict[str, list[int]]:
    """Tokenize text and return a dictionary containing input and labels.

    The output should correspond to that of
    `torchtune.datasets.TextCompletionDataset._prepare_sample`.

    Args:
        text (str):
            The input text to be tokenized.
        add_eos (bool, optional):
            Whether to add an end-of-sequence token. Defaults to False.

    Returns:
        dict[str, list[int]]:
            A dictionary with two keys:
                - "tokens": A list of token IDs representing the input text.
                - "labels": A list of token IDs used as labels, identical to the tokens.
    """
    tokens = tokenizer.value.encode(text=text, add_bos=True, add_eos=add_eos)

    # Truncate if needed, but don't coerce EOS id
    if tokenizer.value.max_seq_len is not None:
        tokens = truncate(tokens, tokenizer.value.max_seq_len - 1)

    # No need to offset labels by 1 - happens in the recipe
    labels = tokens.copy()

    return {"tokens": tokens, "labels": labels}


def exact_deduplication(args: argparse.Namespace) -> int:
    """Perform (normalized) exact deduplication on a dataset loaded from BigQuery.

    Args:
        args (argparse.Namespace):
            A namespace object containing the following attributes:
                - source (str): The source path of the BigQuery dataset.
                - destination (str): The destination path to a parquet file.
                - normal_form (str): The normalization form to be used.

    Returns:
        int:
            The count of deduplicated records.
    """

    df = spark.read.format("bigquery").load(args.source).cache()

    df_dedup = (
        df.withColumn(
            "text_simple", simplify(F.col("text"), F.lit(args.normal_form))
        ).dropDuplicates(subset=["text_simple"])
    ).cache()

    df_dedup.write.parquet(args.destination)

    return df_dedup.count()


def fuzzy_deduplication(args: argparse.Namespace) -> int:
    """Perform fuzzy deduplication on a dataset using MinHash and connected components.

    Uses word-level shingling to represent texts. After pairwise similarities have been
    calculated, pairs are clustered by computing the connected components of their graph
    and one random text from each cluster is kept.

    For example, if A-B and B-C are duplicate pairs, then we will have the A-B-C cluster,
    and one of the three will be retained.


    Args:
        args (argparse.Namespace):
            A namespace object containing the following attributes:
                - source (str): Path to the input parquet file.
                - destination (str): Path to the output parquet file.
                - ngram_length (int): Length of n-grams for shingling.
                - minhash_permutations (int): Number of hash tables for MinHashLSH.
                - similarity_threshold (float): Threshold for approximate similarity join.

    Returns:
        int:
            The count of deduplicated records.
    """

    df = spark.read.parquet(args.source).cache()
    cv = CountVectorizer(inputCol="shingles", outputCol="features")
    df = df.withColumn(
        "shingles", shingle(F.col("text_simple"), F.lit(args.ngram_length))
    ).cache()
    vectorizer = cv.fit(df)
    features = (
        vectorizer.transform(df)
        .filter(non_zero(F.col("features")))["id", "features"]
        .cache()
    )
    mh = MinHashLSH(
        inputCol="features", outputCol="hashes", numHashTables=args.minhash_permutations
    )
    model = mh.fit(features)

    pairs = (
        model.approxSimilarityJoin(
            features, features, threshold=args.similarity_threshold, distCol="dist"
        )
        .filter("dist != 0")
        .withColumns({"id_a": F.col("datasetA.id"), "id_b": F.col("datasetB.id")})[
            ["id_a", "id_b", "dist"]
        ]
        .cache()
    )

    clusters = (
        GraphFrame(
            pairs[["id_a"]].distinct().withColumnRenamed("id_a", "id"),
            pairs[["id_a", "id_b"]].withColumnsRenamed({"id_a": "src", "id_b": "dst"}),
        )
        .connectedComponents(algorithm="graphx")
        .cache()
    )
    keep = clusters.groupby("component").agg(F.any_value("id").alias("id"))
    remove = clusters.join(keep, on="id", how="left_anti")
    df_dedup = df.join(remove, on="id", how="left_anti").cache()
    df_dedup.write.parquet(args.destination)

    return df_dedup.count()


def tokenization(args: argparse.Namespace) -> int:
    """Tokenize text data using a torchtune tokenizer.

    Args:
        args (argparse.Namespace):
            The arguments containing the following attributes:
                - tokenizer_config (str): Path to the tokenizer configuration file.
                - source (str): Path to the source Parquet file containing text data.
                - destination (str): Path to the destination Parquet file to write tokenized data.
    Returns:
        int:
            The number of rows in the tokenized DataFrame.
    """

    cfg = OmegaConf.create(args.tokenizer_config)
    # my-bucket, dir/.../file.py <- gs://my-bucket/dir/.../file.py
    uri = urlparse(cfg.tokenizer.path)
    bucket_name, filepath = uri.netloc, uri.path.lstrip("/")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Client().bucket(bucket_name).blob(filepath).download_to_filename(filepath)

    global tokenizer  # pylint: disable=global-statement
    tokenizer = spark.sparkContext.broadcast(
        torchtune.config.instantiate(cfg.tokenizer)
    )

    df = spark.read.parquet(args.source)
    df_tokens = df.withColumn("tokens", tokenize(F.col("text")))[
        ["id", "tokens"]
    ].cache()
    df_tokens.write.parquet(args.destination)

    return df_tokens.count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--source")
    parser.add_argument("--normal-form")
    parser.add_argument("--ngram-length", type=int)
    parser.add_argument("--minhash-permutations", type=int)
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--tokenizer-config", type=str)
    parser.add_argument("--destination")
    args_ = parser.parse_args()

    if args_.command == "exact-deduplication":
        tot_num_rows = exact_deduplication(args_)
    elif args_.command == "fuzzy-deduplication":
        tot_num_rows = fuzzy_deduplication(args_)
    elif args_.command == "tokenization":
        tot_num_rows = tokenization(args_)
    else:
        sys.exit(1)

    # To enrich KFP metadata later on
    pq.write_metadata(
        pq.ParquetDataset(args_.destination).schema.with_metadata(
            {"tot_num_rows": str(tot_num_rows)}
        ),
        os.path.join(args_.destination, "_metadata"),
    )

    spark.stop()
