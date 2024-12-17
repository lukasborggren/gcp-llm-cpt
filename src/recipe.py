"""Custom torchtune training recipe.

Built on: https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py
"""

import argparse
import json
import os
import site
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Tuple
from unittest.mock import patch

import torch
from dataflux_pytorch.dataflux_checkpoint import DatafluxCheckpoint
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import save as torch_save
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed

# HACK: workaround since the authors have made the recipes module unimportable
try:
    from recipes.full_finetune_distributed import FullFinetuneRecipeDistributed
except ModuleNotFoundError:
    sys.path.insert(
        0, os.path.abspath(os.path.join(site.getsitepackages()[0], "recipes"))
    )
    from full_finetune_distributed import FullFinetuneRecipeDistributed

log = utils.get_logger("DEBUG")


class GCSParquetDataset(Dataset):
    """Class for loading a Parquet dataset from GCS."""

    def __init__(
        self,
        data_uri: str,
        column: str = "tokens",
        split: str = "train",
        **load_dataset_kwargs: dict[str, Any]
    ) -> None:
        self._data = load_dataset(
            "parquet",
            data_files=os.path.join(data_uri, "*.parquet"),
            split=split,
            **load_dataset_kwargs,
        )
        self._column = column

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        tokens = self._data[index][self._column][self._column]
        labels = tokens.copy()
        return {"tokens": tokens, "labels": labels}


class CheckpointHelper:
    """A wrapper class around DatafluxCheckpoint that allows the checkpoint
    save to be called asynchronously.

    Adapted from:
        https://github.com/GoogleCloudPlatform/gcs-connector-for-pytorch/blob/main/demo/checkpointing/train.py
    """

    def __init__(
        self,
        bucket_name: str,
        project_name: str = os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    ) -> None:
        self._ckpt = DatafluxCheckpoint(project_name, bucket_name)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load(self, path: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        with self._ckpt.reader(path) as reader:
            return torch.load(reader, **kwargs)

    def save_config(self, path: Path, config_: dict[str, Any]) -> None:
        filepath = Path.joinpath(path, "config.json")
        blob = self._ckpt.bucket.blob(str(filepath))
        blob.upload_from_string(
            data=json.dumps(config_), content_type="application/json"
        )

    def save(
        self, state_dict: dict[str, torch.Tensor], path: Path, use_async: bool = False
    ) -> None:
        def _save() -> None:
            with self._ckpt.writer(str(path)) as writer:
                torch_save(state_dict, writer)

        if use_async:
            self._executor.submit(_save)
        else:
            _save()

    def getsize(self, path: Path) -> int:
        return self._ckpt.bucket.get_blob(str(path)).size

    def teardown(self) -> None:
        self._executor.shutdown(wait=True)


class CustomRecipe(FullFinetuneRecipeDistributed):
    """Custom recipe that slightly modifies torchtune's `FullFinetuneRecipeDistributed`."""

    def __init__(self, cfg: DictConfig):
        self._ckpt = CheckpointHelper(cfg.bucket_name)
        super().__init__(cfg)

    def _setup_data(
        self, cfg_dataset: DictConfig, shuffle: bool, batch_size: int, collate_fn: str
    ) -> Tuple[DistributedSampler, DataLoader]:
        """Identical to inherited method except for dataset init."""
        world_size, rank = training.get_world_size_and_rank()
        ds = GCSParquetDataset(cfg_dataset.uri)

        # NOTE: From here on, copy-paste from parent recipe class
        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = DistributedSampler(  # type: ignore
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )
        dataloader = DataLoader(  # type: ignore
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(  # type: ignore
                    collate_fn,  # type: ignore
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not cfg_dataset.packed
                else padded_collate_packed
            ),
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        """Monkey patching of inherited method to ensure that model config is saved to GCS."""
        with patch(
            "torchtune.training.checkpointing._checkpointer.save_config",
            self._ckpt.save_config,
        ):
            return super().load_checkpoint(cfg_checkpointer)

    def save_checkpoint(self, epoch: int) -> None:
        """Monkey patching of inherited method to ensure that model checkpoints are saved to GCS."""
        with patch(
            "torchtune.training.checkpointing._checkpointer.torch.save", self._ckpt.save
        ):
            with patch(
                "torchtune.training.checkpointing._checkpointer.os.path.getsize",
                self._ckpt.getsize,
            ):
                super().save_checkpoint(epoch)


def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config("CustomRecipe", cfg)
    for k, c in os.environ.items():
        log.debug("%s: %s", k, c)

    recipe = CustomRecipe(cfg)
    recipe.setup(cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    recipe_main(OmegaConf.create(parser.parse_args().config))
