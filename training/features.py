"""
FH-NNUE Feature Mapping and Indexing (Python Reference)

Must match `src/nnue/features.rs` identically.
"""

NUM_KING_BUCKETS = 32
NUM_PIECE_CATEGORIES = 11
NUM_SQUARES = 64
NUM_FEATURES = NUM_KING_BUCKETS * NUM_PIECE_CATEGORIES * NUM_SQUARES  # 22,528
NUM_REL_DISPLACEMENTS = 15 * 15  # 225
NUM_KING_ZONES = 4

# Relative piece category indices:
# Friendly: P=0, N=1, B=2, R=3, Q=4
# Enemy:    P=5, N=6, B=7, R=8, Q=9, K=10

PIECE_NAMES = ["p", "n", "b", "r", "q", "k"]


def square_to_rank_file(sq: int):
    return sq // 8, sq % 8


def rank_file_to_square(rank: int, file: int) -> int:
    return rank * 8 + file


def flip_vertical(sq: int) -> int:
    return sq ^ 56


def flip_horizontal(sq: int) -> int:
    return sq ^ 7


def orient_square(sq: int, perspective: int, is_mirrored_h: bool) -> int:
    """
    perspective: 0 for White, 1 for Black.
    """
    if perspective == 0:  # White
        return flip_horizontal(sq) if is_mirrored_h else sq
    else:  # Black
        return sq ^ 63 if is_mirrored_h else flip_vertical(sq)


def get_king_orientation(king_sq: int, perspective: int):
    """
    Returns (canonical_king_sq, king_bucket, is_mirrored_h).
    """
    oriented_sq = king_sq if perspective == 0 else flip_vertical(king_sq)
    rank, file = square_to_rank_file(oriented_sq)
    is_mirrored_h = file >= 4

    canon_sq = flip_horizontal(oriented_sq) if is_mirrored_h else oriented_sq
    c_rank, c_file = square_to_rank_file(canon_sq)
    assert c_file < 4, f"Canonical king file must be < 4, got {c_file}"

    king_bucket = c_rank * 4 + c_file
    return canon_sq, king_bucket, is_mirrored_h


def get_king_zone(canon_king_sq: int) -> int:
    c_rank, c_file = square_to_rank_file(canon_king_sq)
    if c_rank <= 1:
        return 0 if c_file <= 1 else 1
    elif c_rank <= 3:
        return 2
    else:
        return 3


def get_relative_displacement(canon_king_sq: int, canon_piece_sq: int) -> int:
    k_rank, k_file = square_to_rank_file(canon_king_sq)
    p_rank, p_file = square_to_rank_file(canon_piece_sq)
    delta_r = (p_rank - k_rank) + 7
    delta_f = (p_file - k_file) + 7
    assert 0 <= delta_r < 15 and 0 <= delta_f < 15
    return delta_r * 15 + delta_f


def get_relative_piece(piece_char: str, piece_color: int, perspective: int):
    """
    Returns relative piece category index (0..10), or None if friendly king.
    """
    piece_lower = piece_char.lower()
    is_friendly = (piece_color == perspective)

    if is_friendly:
        mapping = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4}
        return mapping.get(piece_lower, None)  # King is None
    else:
        mapping = {"p": 5, "n": 6, "b": 7, "r": 8, "q": 9, "k": 10}
        return mapping.get(piece_lower, None)


def feature_index(king_bucket: int, rel_piece: int, canon_sq: int) -> int:
    return king_bucket * (NUM_PIECE_CATEGORIES * NUM_SQUARES) + rel_piece * NUM_SQUARES + canon_sq


def extract_features_from_fen(fen: str, perspective: int):
    """
    Extract active feature indices for the given perspective from a FEN string.
    Returns list of feature indices.
    """
    parts = fen.split()
    board_part = parts[0]

    pieces = []
    king_sqs = {0: None, 1: None}

    rank = 7
    file = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank_file_to_square(rank, file)
            color = 0 if ch.isupper() else 1
            if ch.lower() == "k":
                king_sqs[color] = sq
            pieces.append((sq, ch.lower(), color))
            file += 1

    king_sq = king_sqs[perspective]
    assert king_sq is not None, "King not found"

    canon_king_sq, bucket, is_mirrored_h = get_king_orientation(king_sq, perspective)

    features = []
    for sq, piece_char, color in pieces:
        rel_piece = get_relative_piece(piece_char, color, perspective)
        if rel_piece is not None:
            c_sq = orient_square(sq, perspective, is_mirrored_h)
            idx = feature_index(bucket, rel_piece, c_sq)
            features.append(idx)

    return features, canon_king_sq, bucket


if __name__ == "__main__":
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    w_feats, w_k, w_b = extract_features_from_fen(startpos, 0)
    b_feats, b_k, b_b = extract_features_from_fen(startpos, 1)

    print(f"White active features: {len(w_feats)}, king bucket: {w_b}")
    print(f"Black active features: {len(b_feats)}, king bucket: {b_b}")
    assert len(w_feats) == 31
    assert len(b_feats) == 31
    assert all(0 <= f < NUM_FEATURES for f in w_feats)
    assert all(0 <= f < NUM_FEATURES for f in b_feats)
    print("Features self-test passed successfully!")
