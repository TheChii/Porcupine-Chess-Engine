"""
FH-NNUE PyTorch Model Architecture

Supports:
- B0: Baseline HalfKAv2
- A1: Horizontally canonicalized HalfKAv2
- A2: Global + Local factorized weights
- A3: Global + Relative-displacement + Zone + Local factorized weights
- A4: Optional PSQT skip connection
- Weight folding into flat [22528, 512] feature table for production inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import (
    NUM_KING_BUCKETS,
    NUM_PIECE_CATEGORIES,
    NUM_SQUARES,
    NUM_FEATURES,
    NUM_REL_DISPLACEMENTS,
    NUM_KING_ZONES,
    get_king_zone,
    get_relative_displacement,
)

TRANSFORMER_WIDTH = 512
DENSE1_IN = TRANSFORMER_WIDTH * 2  # 1,024
DENSE1_OUT = 128
DENSE2_OUT = 0
PSQT_CHANNELS = 8


def screlu(x: torch.Tensor) -> torch.Tensor:
    """SCReLU activation: clamp(x, 0, 1)^2."""
    clamped = torch.clamp(x, 0.0, 1.0)
    return clamped * clamped


class FHNNUE(nn.Module):
    def __init__(
        self,
        arch_type: str = "A3",  # B0, A1, A2, A3
        has_psqt: bool = True,
    ):
        super().__init__()
        self.arch_type = arch_type
        self.has_psqt = has_psqt

        # Feature Transformer Weights
        if arch_type in ("B0", "A1"):
            # Flat unfactorized table: [22528, 512]
            self.flat_weights = nn.Parameter(torch.randn(NUM_FEATURES, TRANSFORMER_WIDTH) * 0.01)
        elif arch_type == "A2":
            # Global [11, 64, 512] + Local [32, 11, 64, 512]
            self.w_global = nn.Parameter(torch.randn(NUM_PIECE_CATEGORIES, NUM_SQUARES, TRANSFORMER_WIDTH) * 0.01)
            self.w_local = nn.Parameter(torch.randn(NUM_KING_BUCKETS, NUM_PIECE_CATEGORIES, NUM_SQUARES, TRANSFORMER_WIDTH) * 0.005)
        elif arch_type == "A3":
            # Global + Rel + Zone + Local
            self.w_global = nn.Parameter(torch.randn(NUM_PIECE_CATEGORIES, NUM_SQUARES, TRANSFORMER_WIDTH) * 0.01)
            self.w_rel = nn.Parameter(torch.randn(NUM_PIECE_CATEGORIES, NUM_REL_DISPLACEMENTS, TRANSFORMER_WIDTH) * 0.005)
            self.w_zone = nn.Parameter(torch.randn(NUM_KING_ZONES, NUM_PIECE_CATEGORIES, NUM_SQUARES, TRANSFORMER_WIDTH) * 0.005)
            self.w_local = nn.Parameter(torch.randn(NUM_KING_BUCKETS, NUM_PIECE_CATEGORIES, NUM_SQUARES, TRANSFORMER_WIDTH) * 0.002)
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        # Precompute static indices for vectorized folding
        zones = torch.zeros(NUM_KING_BUCKETS, dtype=torch.long)
        rel_disp = torch.zeros(NUM_KING_BUCKETS, NUM_SQUARES, dtype=torch.long)
        for k in range(NUM_KING_BUCKETS):
            c_rank = k // 4
            c_file = k % 4
            canon_king_sq = c_rank * 8 + c_file
            zones[k] = get_king_zone(canon_king_sq)
            for s in range(NUM_SQUARES):
                rel_disp[k, s] = get_relative_displacement(canon_king_sq, s)

        self.register_buffer("zones", zones)
        self.register_buffer("rel_disp", rel_disp)

        # Feature biases: [512], centered at 0.5 to keep initial activations in active gradient zone
        self.feature_biases = nn.Parameter(torch.full((TRANSFORMER_WIDTH,), 0.5))

        # Optional PSQT Skip Connection
        if has_psqt:
            self.psqt_weights = nn.Parameter(torch.randn(NUM_FEATURES, PSQT_CHANNELS) * 0.01)
            self.psqt_biases = nn.Parameter(torch.zeros(PSQT_CHANNELS))
            self.psqt_head = nn.Linear(PSQT_CHANNELS, 1, bias=False)
            nn.init.normal_(self.psqt_head.weight, std=0.05)
        else:
            self.psqt_weights = None
            self.psqt_biases = None
            self.psqt_head = None

        # Dense Evaluation Head (1024 -> 128 -> 1)
        self.dense1 = nn.Linear(DENSE1_IN, DENSE1_OUT)
        self.output = nn.Linear(DENSE1_OUT, 1)

        nn.init.kaiming_normal_(self.dense1.weight, nonlinearity="relu")
        nn.init.constant_(self.dense1.bias, 0.2)
        nn.init.normal_(self.output.weight, std=0.05)
        nn.init.constant_(self.output.bias, 0.0)

    def fold_weights(self) -> torch.Tensor:
        """
        Vectorized folding of factorized weights into flat [22528, 512] feature table.
        W_flat(k, p, s) = W_global(p, s) + W_rel(p, Δr, Δf) + W_zone(Z(k), p, s) + W_local(k, p, s)
        """
        if self.arch_type in ("B0", "A1"):
            return self.flat_weights

        if self.arch_type == "A2":
            w = self.w_global.unsqueeze(0) + self.w_local
            return w.reshape(NUM_FEATURES, TRANSFORMER_WIDTH)

        if self.arch_type == "A3":
            # w_global: [1, 11, 64, 512]
            w_glob = self.w_global.unsqueeze(0)
            # w_rel: [11, 225, 512] -> [11, 32, 64, 512] -> permute to [32, 11, 64, 512]
            w_rel_k = self.w_rel[:, self.rel_disp].permute(1, 0, 2, 3)
            # w_zone: [4, 11, 64, 512] -> [32, 11, 64, 512]
            w_zone_k = self.w_zone[self.zones]
            # w_local: [32, 11, 64, 512]
            w = w_glob + w_rel_k + w_zone_k + self.w_local
            return w.reshape(NUM_FEATURES, TRANSFORMER_WIDTH)

        raise ValueError(f"Unknown arch_type: {self.arch_type}")

    def forward_from_features(self, stm_features_list, nstm_features_list, flat_weights: torch.Tensor = None):
        """
        Inference from sparse active feature index lists (batch size = len(stm_features_list)).
        """
        if flat_weights is None:
            flat_weights = self.fold_weights()

        batch_size = len(stm_features_list)
        stm_accs = torch.zeros(batch_size, TRANSFORMER_WIDTH, device=flat_weights.device)
        nstm_accs = torch.zeros(batch_size, TRANSFORMER_WIDTH, device=flat_weights.device)

        stm_psqt = torch.zeros(batch_size, PSQT_CHANNELS, device=flat_weights.device) if self.has_psqt else None
        nstm_psqt = torch.zeros(batch_size, PSQT_CHANNELS, device=flat_weights.device) if self.has_psqt else None

        for b in range(batch_size):
            stm_accs[b] = self.feature_biases + flat_weights[stm_features_list[b]].sum(dim=0)
            nstm_accs[b] = self.feature_biases + flat_weights[nstm_features_list[b]].sum(dim=0)

            if self.has_psqt:
                stm_psqt[b] = self.psqt_biases + self.psqt_weights[stm_features_list[b]].sum(dim=0)
                nstm_psqt[b] = self.psqt_biases + self.psqt_weights[nstm_features_list[b]].sum(dim=0)

        # Activations
        stm_act = screlu(stm_accs)
        nstm_act = screlu(nstm_accs)
        combined = torch.cat([stm_act, nstm_act], dim=-1)

        d1 = screlu(self.dense1(combined))
        out = self.output(d1)

        if self.has_psqt:
            psqt_diff = stm_psqt - nstm_psqt
            psqt_out = self.psqt_head(psqt_diff)
            out = out + psqt_out

        return out

    def forward_batched(
        self,
        stm_indices: torch.Tensor,
        stm_offsets: torch.Tensor,
        nstm_indices: torch.Tensor,
        nstm_offsets: torch.Tensor,
        flat_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        High-throughput vectorized inference using F.embedding_bag.
        Eliminates per-sample Python loops and executes entirely in CUDA kernels.
        """
        if flat_weights is None:
            flat_weights = self.fold_weights()

        stm_accs = F.embedding_bag(stm_indices, flat_weights, offsets=stm_offsets, mode="sum") + self.feature_biases
        nstm_accs = F.embedding_bag(nstm_indices, flat_weights, offsets=nstm_offsets, mode="sum") + self.feature_biases

        stm_act = screlu(stm_accs)
        nstm_act = screlu(nstm_accs)
        combined = torch.cat([stm_act, nstm_act], dim=-1)

        d1 = screlu(self.dense1(combined))
        out = self.output(d1)

        if self.has_psqt:
            stm_psqt = F.embedding_bag(stm_indices, self.psqt_weights, offsets=stm_offsets, mode="sum") + self.psqt_biases
            nstm_psqt = F.embedding_bag(nstm_indices, self.psqt_weights, offsets=nstm_offsets, mode="sum") + self.psqt_biases
            psqt_diff = stm_psqt - nstm_psqt
            psqt_out = self.psqt_head(psqt_diff)
            out = out + psqt_out

        return out



if __name__ == "__main__":
    for arch in ["B0", "A1", "A2", "A3"]:
        model = FHNNUE(arch_type=arch, has_psqt=True)
        folded = model.fold_weights()
        assert folded.shape == (22528, 512), f"Folded shape mismatch for {arch}: {folded.shape}"
        print(f"Arch {arch} folding verified: shape {folded.shape}")

    # Forward pass test
    model_a3 = FHNNUE(arch_type="A3", has_psqt=True)
    f_w = [10, 25, 100, 500]
    f_b = [15, 30, 110, 505]
    out = model_a3.forward_from_features([f_w], [f_b])
    print(f"Sample forward pass output: {out.item():.4f}")
    print("Model tests passed!")
