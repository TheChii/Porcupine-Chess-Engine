//! NNUE feature mapping module: canonical king buckets, orientation, and indexing.
//!
//! # Architecture: HalfKAv2 with Horizontal Canonicalization
//!
//! - Perspectives: Evaluated from both White and Black perspectives.
//! - Canonical King Buckets: The own king square is the bucket.
//!   The board is horizontally mirrored (sq ^ 7) if the own king is on files E-H (>= 4),
//!   so the own king always occupies files A-D (0-3).
//!   For Black, the board is vertically flipped (sq ^ 56) first, then horizontally
//!   mirrored if needed (sq ^ 63).
//!   This yields 32 canonical king squares (4 files x 8 ranks).
//! - Relative Piece Categories:
//!   Friendly: Pawn (0), Knight (1), Bishop (2), Rook (3), Queen (4)
//!   Enemy: Pawn (5), Knight (6), Bishop (7), Rook (8), Queen (9), King (10)
//!   (Friendly king is represented by the king bucket itself).
//! - Ordinary feature indices:
//!   `32 buckets * 11 categories * 64 squares = 22,528` indices per perspective.

use crate::types::{Board, CastleRights, Color, Piece, Square};

/// Total number of canonical king buckets (files A-D x ranks 1-8).
pub const NUM_KING_BUCKETS: usize = 32;

/// Number of relative piece categories (5 friendly + 6 enemy).
pub const NUM_PIECE_CATEGORIES: usize = 11;

/// Number of squares on the chess board.
pub const NUM_SQUARES: usize = 64;

/// Total number of ordinary HalfKAv2 features per perspective.
pub const NUM_FEATURES: usize = NUM_KING_BUCKETS * NUM_PIECE_CATEGORIES * NUM_SQUARES; // 22,528

/// Number of relative displacement offsets (delta_r in -7..7, delta_f in -7..7).
pub const NUM_REL_DISPLACEMENTS: usize = 15 * 15; // 225

/// Number of coarse king zones for factorized transformer.
pub const NUM_KING_ZONES: usize = 4;

/// Relative piece category index (0..10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RelativePiece {
    FriendlyPawn = 0,
    FriendlyKnight = 1,
    FriendlyBishop = 2,
    FriendlyRook = 3,
    FriendlyQueen = 4,
    EnemyPawn = 5,
    EnemyKnight = 6,
    EnemyBishop = 7,
    EnemyRook = 8,
    EnemyQueen = 9,
    EnemyKing = 10,
}

impl RelativePiece {
    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Convert a piece and its color relative to perspective into RelativePiece.
    /// Returns None for friendly king since it forms the king bucket.
    #[inline(always)]
    pub const fn from_piece(piece: Piece, piece_color: Color, perspective: Color) -> Option<Self> {
        let is_friendly = (piece_color as u8) == (perspective as u8);
        if is_friendly {
            match piece {
                Piece::Pawn => Some(RelativePiece::FriendlyPawn),
                Piece::Knight => Some(RelativePiece::FriendlyKnight),
                Piece::Bishop => Some(RelativePiece::FriendlyBishop),
                Piece::Rook => Some(RelativePiece::FriendlyRook),
                Piece::Queen => Some(RelativePiece::FriendlyQueen),
                Piece::King => None, // King bucket
            }
        } else {
            match piece {
                Piece::Pawn => Some(RelativePiece::EnemyPawn),
                Piece::Knight => Some(RelativePiece::EnemyKnight),
                Piece::Bishop => Some(RelativePiece::EnemyBishop),
                Piece::Rook => Some(RelativePiece::EnemyRook),
                Piece::Queen => Some(RelativePiece::EnemyQueen),
                Piece::King => Some(RelativePiece::EnemyKing),
            }
        }
    }
}

/// Perspective-relative orientation and canonicalization mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerspectiveOrientation {
    pub perspective: Color,
    pub king_sq: Square,
    pub canonical_king_sq: Square,
    pub king_bucket: usize,
    pub is_mirrored_h: bool,
}

impl Default for PerspectiveOrientation {
    fn default() -> Self {
        Self {
            perspective: Color::White,
            king_sq: Square::E1,
            canonical_king_sq: Square::D1,
            king_bucket: 3,
            is_mirrored_h: true,
        }
    }
}

impl PerspectiveOrientation {
    /// Compute the orientation context for a perspective given the board.
    #[inline]
    pub fn new(board: &Board, perspective: Color) -> Self {
        let king_sq = board.king_square(perspective);
        Self::from_king_square(king_sq, perspective)
    }

    /// Compute the orientation context given perspective and king square.
    #[inline]
    pub fn from_king_square(king_sq: Square, perspective: Color) -> Self {
        let oriented_king_sq = match perspective {
            Color::White => king_sq,
            Color::Black => king_sq.flip_vertical(),
        };

        let file = oriented_king_sq.file().index();
        let is_mirrored_h = file >= 4;

        let canonical_king_sq = if is_mirrored_h {
            oriented_king_sq.flip_horizontal()
        } else {
            oriented_king_sq
        };

        let c_file = canonical_king_sq.file().index() as usize;
        let c_rank = canonical_king_sq.rank().index() as usize;
        debug_assert!(c_file < 4);
        debug_assert!(c_rank < 8);

        let king_bucket = c_rank * 4 + c_file;
        debug_assert!(king_bucket < NUM_KING_BUCKETS);

        Self {
            perspective,
            king_sq,
            canonical_king_sq,
            king_bucket,
            is_mirrored_h,
        }
    }

    /// Map any board square to its canonical square for this perspective.
    #[inline(always)]
    pub fn map_square(&self, sq: Square) -> Square {
        match (self.perspective, self.is_mirrored_h) {
            (Color::White, false) => sq,
            (Color::White, true) => sq.flip_horizontal(),
            (Color::Black, false) => sq.flip_vertical(),
            (Color::Black, true) => unsafe { Square::from_index_unchecked(sq.index() ^ 63) },
        }
    }

    /// Map canonical square and relative piece to feature index.
    #[inline(always)]
    pub fn feature_index(&self, rel_piece: RelativePiece, canonical_sq: Square) -> usize {
        debug_assert!(canonical_sq.index() < 64);
        self.king_bucket * (NUM_PIECE_CATEGORIES * NUM_SQUARES)
            + rel_piece.index() * NUM_SQUARES
            + canonical_sq.index() as usize
    }

    /// Map coarse zone for the king bucket (0..3).
    #[inline(always)]
    pub fn king_zone(&self) -> usize {
        let c_file = self.canonical_king_sq.file().index();
        let c_rank = self.canonical_king_sq.rank().index();
        if c_rank <= 1 {
            if c_file <= 1 {
                0 // Castled corner
            } else {
                1 // Queenside / center
            }
        } else if c_rank <= 3 {
            2 // Middle ranks
        } else {
            3 // Enemy half / advanced
        }
    }

    /// Compute relative displacement coordinates (delta_r, delta_f) mapped to index [0..224].
    #[inline(always)]
    pub fn relative_displacement_index(&self, canonical_sq: Square) -> usize {
        let k_rank = self.canonical_king_sq.rank().index() as i8;
        let k_file = self.canonical_king_sq.file().index() as i8;
        let p_rank = canonical_sq.rank().index() as i8;
        let p_file = canonical_sq.file().index() as i8;

        let delta_r = (p_rank - k_rank) + 7; // [-7..7] -> [0..14]
        let delta_f = (p_file - k_file) + 7; // [-7..7] -> [0..14]

        debug_assert!(delta_r >= 0 && delta_r < 15);
        debug_assert!(delta_f >= 0 && delta_f < 15);

        (delta_r as usize) * 15 + (delta_f as usize)
    }

    /// Mirror castling rights to match horizontal reflection.
    /// In standard chess, horizontal reflection swaps kingside (H-rook) and queenside (A-rook).
    #[inline]
    pub fn canonical_castling_rights(&self, cr: CastleRights) -> CastleRights {
        if !self.is_mirrored_h {
            return cr;
        }

        // Swap white kingside (bit 0) and queenside (bit 1)
        // Swap black kingside (bit 2) and queenside (bit 3)
        let bits = cr.bits();
        let w_ks = (bits & 0b0001) != 0;
        let w_qs = (bits & 0b0010) != 0;
        let b_ks = (bits & 0b0100) != 0;
        let b_qs = (bits & 0b1000) != 0;

        let mut mirrored = 0u8;
        if w_ks {
            mirrored |= 0b0010;
        }
        if w_qs {
            mirrored |= 0b0001;
        }
        if b_ks {
            mirrored |= 0b1000;
        }
        if b_qs {
            mirrored |= 0b0100;
        }

        CastleRights::from_bits(mirrored)
    }
}

/// An active feature represented by its global feature index.
pub type FeatureIndex = usize;

/// Enumerate all active feature indices for a perspective on a given board.
/// Max active features on a chess board = 30 (excluding friendly king and at least one enemy piece or pawns).
pub fn enumerate_features(board: &Board, perspective: Color) -> smallvec::SmallVec<[FeatureIndex; 32]> {
    let orientation = PerspectiveOrientation::new(board, perspective);
    let mut features = smallvec::SmallVec::new();

    for &piece in &Piece::ALL {
        for &color in &Color::ALL {
            if let Some(rel_piece) = RelativePiece::from_piece(piece, color, perspective) {
                let bb = board.piece_color_bb(piece, color);
                for sq in bb {
                    let c_sq = orientation.map_square(sq);
                    let idx = orientation.feature_index(rel_piece, c_sq);
                    features.push(idx);
                }
            }
        }
    }

    features
}

/// A feature change containing features to remove and features to add.
#[derive(Debug, Clone, Default)]
pub struct FeatureDiff {
    pub removed: smallvec::SmallVec<[FeatureIndex; 4]>,
    pub added: smallvec::SmallVec<[FeatureIndex; 4]>,
}

/// Compute feature diffs caused by a move for a given perspective orientation.
///
/// NOTE: If the perspective's own king moves, its orientation/bucket changes,
/// so incremental update is NOT applicable; a full refresh must be performed instead.
/// This function assumes the perspective's own king DID NOT move.
pub fn compute_feature_diff_non_king(
    orientation: &PerspectiveOrientation,
    moved_piece: Piece,
    moved_color: Color,
    from: Square,
    to: Square,
    captured: Option<(Piece, Color)>,
    promotion: Option<Piece>,
) -> FeatureDiff {
    let mut diff = FeatureDiff::default();

    // 1. Moving piece from old square
    if let Some(rel_moving) = RelativePiece::from_piece(moved_piece, moved_color, orientation.perspective) {
        let c_from = orientation.map_square(from);
        diff.removed.push(orientation.feature_index(rel_moving, c_from));
    }

    // 2. Captured piece on its square (if any)
    if let Some((cap_piece, cap_color)) = captured {
        if let Some(rel_cap) = RelativePiece::from_piece(cap_piece, cap_color, orientation.perspective) {
            let c_to = orientation.map_square(to);
            diff.removed.push(orientation.feature_index(rel_cap, c_to));
        }
    }

    // 3. Piece appearing on destination square (either promoted piece or moved piece)
    let final_piece = promotion.unwrap_or(moved_piece);
    if let Some(rel_final) = RelativePiece::from_piece(final_piece, moved_color, orientation.perspective) {
        let c_to = orientation.map_square(to);
        diff.added.push(orientation.feature_index(rel_final, c_to));
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_king_buckets() {
        for rank in 0..8 {
            for file in 0..8 {
                let sq = Square::from_file_rank(
                    crate::types::File::from_index(file).unwrap(),
                    crate::types::Rank::from_index(rank).unwrap(),
                );
                let w_orient = PerspectiveOrientation::from_king_square(sq, Color::White);
                assert!(w_orient.king_bucket < 32);
                assert!(w_orient.canonical_king_sq.file().index() < 4);

                let b_orient = PerspectiveOrientation::from_king_square(sq, Color::Black);
                assert!(b_orient.king_bucket < 32);
                assert!(b_orient.canonical_king_sq.file().index() < 4);
            }
        }
    }

    #[test]
    fn test_feature_index_bounds() {
        let board = Board::startpos();
        let w_features = enumerate_features(&board, Color::White);
        let b_features = enumerate_features(&board, Color::Black);

        // Standard chess initial position has 16 white pieces + 16 black pieces = 32.
        // Each perspective omits its own king, leaving 31 active features.
        assert_eq!(w_features.len(), 31);
        assert_eq!(b_features.len(), 31);

        for &feat in &w_features {
            assert!(feat < NUM_FEATURES, "Feature index {} >= {}", feat, NUM_FEATURES);
        }
        for &feat in &b_features {
            assert!(feat < NUM_FEATURES, "Feature index {} >= {}", feat, NUM_FEATURES);
        }
    }

    #[test]
    fn test_horizontal_symmetry_mapping() {
        // King on E1 (file 4, rank 0). Mirrored should be D1 (file 3, rank 0).
        let e1 = Square::E1;
        let orient = PerspectiveOrientation::from_king_square(e1, Color::White);
        assert!(orient.is_mirrored_h);
        assert_eq!(orient.canonical_king_sq, Square::D1);
        assert_eq!(orient.king_bucket, 3); // rank 0 * 4 + file 3

        // King on B1 (file 1, rank 0). Not mirrored: B1.
        let b1 = Square::B1;
        let orient_b1 = PerspectiveOrientation::from_king_square(b1, Color::White);
        assert!(!orient_b1.is_mirrored_h);
        assert_eq!(orient_b1.canonical_king_sq, Square::B1);
        assert_eq!(orient_b1.king_bucket, 1);
    }

    #[test]
    fn test_castling_rights_canonicalization() {
        let e1 = Square::E1;
        let orient = PerspectiveOrientation::from_king_square(e1, Color::White);
        assert!(orient.is_mirrored_h);

        // Original: White kingside only (0b0001)
        let cr = CastleRights::WHITE_KINGSIDE;
        let canon_cr = orient.canonical_castling_rights(cr);
        // Mirrored should become White queenside (0b0010)
        assert_eq!(canon_cr, CastleRights::WHITE_QUEENSIDE);

        // Both rights: 0b0011 -> 0b0011
        let cr_both = CastleRights::WHITE;
        assert_eq!(orient.canonical_castling_rights(cr_both), CastleRights::WHITE);
    }
}
