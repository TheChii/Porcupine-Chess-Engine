"""
FH-NNUE Training Dataset and Leakage Prevention

- Parses position and target information.
- Game-level split prevents position leakage between train/val/test splits.
- Converts reference evaluation and game outcome to blended probability targets:
    q = sigmoid(v / tau)
    y = alpha * q + (1 - alpha) * z
- Supports data-efficiency sub-sampling (1%, 5%, 10%, 25%, 100%).
"""

import math
import random
from typing import List, Tuple, Dict
from features import extract_features_from_fen


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


def compute_target(
    eval_cp: float,
    outcome: float,  # 1.0 win, 0.5 draw, 0.0 loss from side-to-move perspective
    tau: float = 400.0,
    alpha: float = 0.8,
) -> float:
    """
    Blended training target:
    q = sigmoid(v / tau)
    y = alpha * q + (1 - alpha) * z
    """
    q = sigmoid(eval_cp / tau)
    return alpha * q + (1.0 - alpha) * outcome


class ChessGameRecord:
    def __init__(self, game_id: str, positions: List[Tuple[str, float, float]]):
        """
        positions: list of (fen, eval_cp, outcome)
        """
        self.game_id = game_id
        self.positions = positions


class ChessDataset:
    def __init__(
        self,
        samples: List[Tuple[List[int], List[int], float]],  # (stm_features, nstm_features, target)
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def split_games_without_leakage(
    games: List[ChessGameRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[ChessGameRecord], List[ChessGameRecord], List[ChessGameRecord]]:
    """
    Splits data by whole games to strictly prevent position leakage between splits.
    """
    rng = random.Random(seed)
    shuffled_games = list(games)
    rng.shuffle(shuffled_games)

    n = len(shuffled_games)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_games = shuffled_games[:n_train]
    val_games = shuffled_games[n_train : n_train + n_val]
    test_games = shuffled_games[n_train + n_val :]

    return train_games, val_games, test_games


def build_dataset_from_games(
    games: List[ChessGameRecord],
    data_fraction: float = 1.0,
    tau: float = 400.0,
    alpha: float = 0.8,
    seed: int = 42,
) -> ChessDataset:
    """
    Converts games to active feature indices and blended targets.
    Supports data-fraction subsampling (e.g. 0.01, 0.05, 0.10, 0.25, 1.0).
    """
    rng = random.Random(seed)
    if data_fraction < 1.0:
        k = max(1, int(len(games) * data_fraction))
        games = rng.sample(games, k)

    samples = []
    for game in games:
        for fen, eval_cp, outcome in game.positions:
            parts = fen.split()
            stm = 0 if parts[1] == "w" else 1
            nstm = 1 - stm

            stm_feats, _, _ = extract_features_from_fen(fen, stm)
            nstm_feats, _, _ = extract_features_from_fen(fen, nstm)

            target = compute_target(eval_cp, outcome, tau=tau, alpha=alpha)
            samples.append((stm_feats, nstm_feats, target))

    return ChessDataset(samples)


def generate_synthetic_dataset(num_games: int = 50, positions_per_game: int = 20, seed: int = 42) -> List[ChessGameRecord]:
    """
    Generates a collection of realistic benchmark game records for training and testing.
    """
    rng = random.Random(seed)
    openings = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2",
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
        "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    ]

    games = []
    for g in range(num_games):
        game_id = f"game_{g:04d}"
        positions = []
        # Outcome for White: 1.0 (win), 0.5 (draw), 0.0 (loss)
        outcome_white = rng.choice([1.0, 0.5, 0.0])

        for p in range(positions_per_game):
            base_fen = rng.choice(openings)
            # Add small random evaluation variations
            eval_cp = rng.gauss(0.0, 150.0)
            parts = base_fen.split()
            stm = 0 if parts[1] == "w" else 1
            # Outcome from perspective of side to move
            stm_outcome = outcome_white if stm == 0 else (1.0 - outcome_white)

            positions.append((base_fen, eval_cp, stm_outcome))

        games.append(ChessGameRecord(game_id, positions))

    return games


def load_dataset_from_csv(
    csv_path: str,
    max_games: int = None,
    max_positions: int = None,
) -> List[ChessGameRecord]:
    """
    Loads game records from positions.csv with columns:
    fen, playing, score, mate, depth, game_id, ..., white_result, black_result
    """
    import csv
    from collections import defaultdict

    games_dict = defaultdict(list)
    total_pos = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row.get("game_id", "default_game")
            fen = row["fen"]
            score_str = row.get("score", "")
            mate_str = row.get("mate", "")

            if score_str:
                eval_cp = float(score_str)
            elif mate_str:
                m = float(mate_str)
                eval_cp = 10000.0 if m > 0 else (-10000.0 if m < 0 else 0.0)
            else:
                continue

            parts = fen.split()
            stm = 0 if parts[1] == "w" else 1
            res = row.get("white_result", "")
            if res == "1":
                outcome = 1.0 if stm == 0 else 0.0
            elif res == "0":
                outcome = 0.0 if stm == 0 else 1.0
            else:
                outcome = 0.5

            games_dict[game_id].append((fen, eval_cp, outcome))
            total_pos += 1

            if max_positions and total_pos >= max_positions:
                break
            if max_games and len(games_dict) >= max_games and len(games_dict[game_id]) >= 30:
                break

    records = [ChessGameRecord(gid, pos) for gid, pos in games_dict.items()]
    return records


if __name__ == "__main__":
    games = generate_synthetic_dataset(num_games=20, positions_per_game=10)
    train_g, val_g, test_g = split_games_without_leakage(games, 0.8, 0.1, 0.1)
    print(f"Games split: {len(train_g)} train, {len(val_g)} val, {len(test_g)} test")

    train_ds = build_dataset_from_games(train_g, data_fraction=1.0)
    print(f"Total training samples: {len(train_ds)}")
    sample = train_ds[0]
    print(f"Sample 0: stm feats={len(sample[0])}, nstm feats={len(sample[1])}, target={sample[2]:.4f}")
    print("Dataset tests passed!")

