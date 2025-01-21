"""Utilities for training pipeline."""

import datetime
import logging
import os
import socket
import subprocess
import time
from typing import Optional

import etcd
from google.cloud import storage
from kfp import dsl
from torch.distributed.elastic.rendezvous.etcd_store import EtcdStore

log = logging.getLogger(__name__)


class CustomEtcdStore(EtcdStore):
    """Overrides torch's `EtcdStore` to fix known bug at rendezvous.

    https://github.com/pytorch/pytorch/issues/132950
    """

    def _try_wait_get(
        self, b64_keys: list[str], override_timeout: Optional[datetime.timedelta] = None
    ) -> Optional[dict[str, bytes]]:
        """Implements fix from: https://github.com/pytorch/pytorch/pull/137056"""
        timeout = self.timeout if override_timeout is None else override_timeout  # type: ignore
        deadline = time.time() + timeout.total_seconds()

        while True:
            # Read whole directory (of keys), filter only the ones waited for
            all_nodes = None
            try:
                all_nodes = self.client.get(key=self.prefix)
                req_nodes = {
                    node.key: node.value
                    for node in all_nodes.children
                    if node.key in b64_keys
                }

                if len(req_nodes) == len(b64_keys):
                    # All keys are available
                    return req_nodes
            except etcd.EtcdKeyNotFound:
                pass

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None

            try:
                index = all_nodes.etcd_index + 1 if all_nodes else 0
                self.client.watch(
                    key=self.prefix,
                    recursive=True,
                    timeout=watch_timeout,
                    index=index,
                )
            except etcd.EtcdWatchTimedOut:
                if time.time() >= deadline:
                    return None
            except etcd.EtcdEventIndexCleared:
                continue


def start_etcd_server(bucket: storage.Bucket) -> None:
    """Starts etcd server on host node and propagates host IP to worker nodes."""
    ip_filename = f"{dsl.get_uri()}/host_ip"

    if os.environ["RANK"] == "0":
        os.environ["HOST_IP"] = socket.gethostbyname(socket.gethostname())
        # Write host IP to GCS for other nodes to read
        bucket.blob(ip_filename).upload_from_string(os.environ["HOST_IP"])

        cmd = f"/tmp/etcd-download/etcd --name s1 --data-dir /tmp/etcd-download/s1  \
            --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://{os.environ['HOST_IP']}:2379 \
            --listen-peer-urls http://0.0.0.0:2380 --initial-advertise-peer-urls http://{os.environ['HOST_IP']}:2380 \
            --initial-cluster s1=http://{os.environ['HOST_IP']}:2380 --initial-cluster-token tkn"

        log.info("Starting etcd server on host node")
        # pylint: disable=consider-using-with
        subprocess.Popen(cmd.split(), env=dict(os.environ, ETCDCTL_API="2"))
        time.sleep(10)  # Wait for etcd server to start in background
    else:
        log.info("Waiting for host IP to be uploaded to GCS")
        for _ in range(60):
            blob = bucket.blob(ip_filename)
            if blob.exists():
                break
            time.sleep(1)

        # Read host IP from GCS
        os.environ["HOST_IP"] = blob.download_as_string().decode()
