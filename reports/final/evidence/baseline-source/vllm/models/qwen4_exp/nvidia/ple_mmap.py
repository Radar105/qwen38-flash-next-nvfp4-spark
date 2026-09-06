# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local GB10 PLE lookup using the original checkpoint's file-backed FP8 rows."""

import json
import math
import struct
from pathlib import Path

import numpy as np
import regex as re
import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class MmapPLEEmbedding(nn.Module):
    """Read only requested rows; preserve the checkpoint's bytes and global scale.

    This local implementation deliberately requires eager TP1 execution. CUDA
    graph capture cannot include its synchronous GPU-to-CPU index transfer.
    """

    def __init__(
        self,
        model_path: str,
        prefix: str,
        num_embeddings: int,
        embedding_dim: int,
        split_ngram_parts: int,
    ) -> None:
        super().__init__()
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.shard_size = math.ceil(num_embeddings / split_ngram_parts)
        self.register_buffer("weight_scale", torch.full((1,), float("nan")))
        root = Path(model_path).resolve()
        index = json.loads((root / "model.safetensors.index.json").read_text())
        # The conditional-generation wrapper remaps this checkpoint prefix.
        if prefix.startswith("language_model.model."):
            prefix = "model.language_model." + prefix.removeprefix(
                "language_model.model."
            )
        pattern = re.compile(re.escape(prefix) + r"\.shard_(\d+)\.weight$")
        entries = {
            name: filename
            for name, filename in index["weight_map"].items()
            if pattern.fullmatch(name)
        }
        if len(entries) != split_ngram_parts:
            raise ValueError("PLE mmap requires every checkpoint embedding shard")
        headers = {}
        self.shards = {}
        for name, filename in entries.items():
            path = (root / filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("PLE shard path escapes model directory")
            if path not in headers:
                with path.open("rb") as handle:
                    length = struct.unpack("<Q", handle.read(8))[0]
                    if length > 100_000_000:
                        raise ValueError("Unexpectedly large safetensors header")
                    headers[path] = (8 + length, json.loads(handle.read(length)))
            start, header = headers[path]
            metadata = header[name]
            match = pattern.fullmatch(name)
            assert match is not None
            shard_id = int(match.group(1))
            rows = min(self.shard_size, num_embeddings - shard_id * self.shard_size)
            if not 0 <= shard_id < split_ngram_parts or rows <= 0:
                raise ValueError("Invalid PLE shard index")
            if metadata["dtype"] != "F8_E4M3":
                raise ValueError("PLE mmap currently requires F8_E4M3 checkpoint rows")
            if metadata["shape"] != [rows, embedding_dim]:
                raise ValueError(f"Invalid PLE shard shape: {name}")
            lo, hi = metadata["data_offsets"]
            if lo < 0 or hi - lo != rows * embedding_dim:
                raise ValueError("Invalid PLE byte range")
            if start + hi > path.stat().st_size:
                raise ValueError("Truncated PLE checkpoint shard")
            self.shards[shard_id] = np.memmap(
                path,
                mode="r",
                dtype=np.uint8,
                offset=start + lo,
                shape=(rows, embedding_dim),
            )
        logger.info(
            "PLE mmap: %d shards, %.3f GiB file-backed FP8; no resident table copy",
            len(self.shards),
            num_embeddings * embedding_dim / 2**30,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PLE mmap requires --enforce-eager")
        if not torch.isfinite(self.weight_scale).all():
            raise RuntimeError("PLE mmap is missing the checkpoint global scale")
        ids = indices.detach().to(device="cpu", dtype=torch.long).numpy().reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.org_vocab_size):
            raise IndexError("PLE lookup index outside vocabulary")
        unique, inverse = np.unique(ids, return_inverse=True)
        packed = np.empty((len(unique), self.embedding_dim), dtype=np.uint8)
        shard_ids = unique // self.shard_size
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            packed[mask] = self.shards[int(shard_id)][unique[mask] % self.shard_size]
        # Transfer packed FP8 bytes, preserving upstream GPU dequantization.
        rows = torch.from_numpy(packed[inverse]).view(torch.float8_e4m3fn)
        return rows.to(indices.device).reshape(*indices.shape, self.embedding_dim)
