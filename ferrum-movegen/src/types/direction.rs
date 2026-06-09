//! Direction type for ray-based movement.

/// Represents the 8 cardinal and diagonal directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum Direction {
    North = 8,
    South = -8,
    East = 1,
    West = -1,
    NorthEast = 9,
    NorthWest = 7,
    SouthEast = -7,
    SouthWest = -9,
}

impl Direction {
    /// All 8 directions.
    pub const ALL: [Direction; 8] = [
        Direction::North, Direction::South, Direction::East, Direction::West,
        Direction::NorthEast, Direction::NorthWest, Direction::SouthEast, Direction::SouthWest,
    ];

    /// Orthogonal directions (rook-like).
    pub const ORTHOGONAL: [Direction; 4] = [
        Direction::North, Direction::South, Direction::East, Direction::West,
    ];

    /// Diagonal directions (bishop-like).
    pub const DIAGONAL: [Direction; 4] = [
        Direction::NorthEast, Direction::NorthWest, Direction::SouthEast, Direction::SouthWest,
    ];

    /// Get the delta value for this direction.
    #[inline(always)]
    pub const fn delta(self) -> i8 {
        self as i8
    }

    /// Get the opposite direction.
    #[inline(always)]
    pub const fn opposite(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            Direction::NorthEast => Direction::SouthWest,
            Direction::NorthWest => Direction::SouthEast,
            Direction::SouthEast => Direction::NorthWest,
            Direction::SouthWest => Direction::NorthEast,
        }
    }

    /// Check if this is a diagonal direction.
    #[inline(always)]
    pub const fn is_diagonal(self) -> bool {
        matches!(self, Direction::NorthEast | Direction::NorthWest | 
                      Direction::SouthEast | Direction::SouthWest)
    }

    /// Check if this is an orthogonal direction.
    #[inline(always)]
    pub const fn is_orthogonal(self) -> bool {
        !self.is_diagonal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::North.opposite(), Direction::South);
        assert_eq!(Direction::NorthEast.opposite(), Direction::SouthWest);
    }

    #[test]
    fn test_direction_delta() {
        assert_eq!(Direction::North.delta(), 8);
        assert_eq!(Direction::South.delta(), -8);
        assert_eq!(Direction::NorthEast.delta(), 9);
    }
}
