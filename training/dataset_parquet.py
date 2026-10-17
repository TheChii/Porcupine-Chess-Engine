"""
Streaming Parquet Dataset for FH-NNUE Training.

Streams positions from a compressed parquet file in chunks, extracting
active feature indices on the fly with minimal RAM footprint (< 500 MB).
"""

import math
import random
import torch
from torch.utils.data import Dataset, IterableDataset
import pyarrow.parquet as pq
from training.features import extract_features_from_fen


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


class ChunkedParquetDataset(IterableDataset):
    """
    High-throughput chunked streaming dataset for large Parquet files.
    Partitions row groups across workers and shuffles locally within chunks.
    """
    def __init__(
        self,
        parquet_path: str,
        tau: float = 600.0,
        rg_slice: slice = None,
        shuffle_chunks: bool = True,
        max_positions: int = None,
    ):
        self.parquet_path = parquet_path
        self.tau = tau
        self.shuffle_chunks = shuffle_chunks
        self.max_positions = max_positions
        
        pf = pq.ParquetFile(parquet_path)
        all_rgs = list(range(pf.num_row_groups))
        if rg_slice is not None:
            self.rgs = all_rgs[rg_slice]
        else:
            self.rgs = all_rgs

        # Estimate total rows in this slice
        total = 0
        for rg_idx in self.rgs:
            total += pf.metadata.row_group(rg_idx).num_rows
        self.total_rows = min(total, max_positions) if max_positions else total

    def __len__(self):
        return self.total_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pf = pq.ParquetFile(self.parquet_path)
        rg_indices = list(self.rgs)

        if self.shuffle_chunks:
            random.shuffle(rg_indices)

        if worker_info is not None:
            # Shard row groups across worker processes
            rg_indices = [rg for i, rg in enumerate(rg_indices) if i % worker_info.num_workers == worker_info.id]

        yielded = 0
        for rg_idx in rg_indices:
            table = pf.read_row_group(rg_idx, columns=["fen", "cp"])
            fens = table["fen"].to_pylist()
            cps = table["cp"].to_pylist()

            row_indices = list(range(len(fens)))
            if self.shuffle_chunks:
                random.shuffle(row_indices)

            for idx in row_indices:
                fen = fens[idx]
                cp = cps[idx]

                # Determine STM from FEN
                parts = fen.split(" ")
                is_black_stm = (len(parts) > 1 and parts[1] == "b")

                if is_black_stm:
                    stm_features = extract_features_from_fen(fen, 1)
                    nstm_features = extract_features_from_fen(fen, 0)
                else:
                    stm_features = extract_features_from_fen(fen, 0)
                    nstm_features = extract_features_from_fen(fen, 1)

                target = sigmoid(float(cp) / self.tau)
                yield stm_features, nstm_features, target

                yielded += 1
                if self.max_positions and yielded >= self.max_positions:
                    return
