//! Polyglot opening book format reader.

use super::zobrist::polyglot_hash;
use crate::types::{Board, Move, Piece};
use movegen::{Square, File, Rank};
use std::fs::File as FsFile;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const ENTRY_SIZE: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct BookEntry {
    pub key: u64,
    pub raw_move: u16,
    pub weight: u16,
    pub learn: u32,
}

impl BookEntry {
    fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self {
            key: u64::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3],
                                     bytes[4], bytes[5], bytes[6], bytes[7]]),
            raw_move: u16::from_be_bytes([bytes[8], bytes[9]]),
            weight: u16::from_be_bytes([bytes[10], bytes[11]]),
            learn: u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        }
    }

    pub fn decode_move(&self) -> (Square, Square, Option<Piece>) {
        let to_file = (self.raw_move & 0x7) as u8;
        let to_rank = ((self.raw_move >> 3) & 0x7) as u8;
        let from_file = ((self.raw_move >> 6) & 0x7) as u8;
        let from_rank = ((self.raw_move >> 9) & 0x7) as u8;
        let promo = ((self.raw_move >> 12) & 0x7) as usize;

        let from = Square::from_file_rank(File::from_index(from_file).unwrap(), Rank::from_index(from_rank).unwrap());
        let to = Square::from_file_rank(File::from_index(to_file).unwrap(), Rank::from_index(to_rank).unwrap());

        let promotion = match promo {
            1 => Some(Piece::Knight), 2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook), 4 => Some(Piece::Queen), _ => None,
        };
        (from, to, promotion)
    }

    pub fn to_chess_move(&self, board: &Board) -> Option<Move> {
        let (from, to, promo) = self.decode_move();
        let actual_to = self.adjust_castling_move(board, from, to);
        
        for m in board.generate_moves().iter() {
            if m.from() == from && m.to() == actual_to {
                if promo.is_some() {
                    if m.flag().promotion_piece() == promo { return Some(m); }
                } else if m.flag().promotion_piece().is_none() {
                    return Some(m);
                }
            }
        }
        None
    }

    fn adjust_castling_move(&self, board: &Board, from: Square, to: Square) -> Square {
        if let Some((piece, _)) = board.piece_at(from) {
            if piece == Piece::King && from.file() == File::E {
                if to.file() == File::H { return Square::from_file_rank(File::G, to.rank()); }
                if to.file() == File::A { return Square::from_file_rank(File::C, to.rank()); }
            }
        }
        to
    }
}

pub struct PolyglotBook {
    data: BookData,
    entry_count: usize,
    pub desc: String,
}

enum BookData { Memory(Vec<BookEntry>), File { path: String } }

impl PolyglotBook {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        let mut file = FsFile::open(path)?;
        let file_size = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;
        
        if file_size % ENTRY_SIZE as u64 != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid book"));
        }
        
        let entry_count = (file_size / ENTRY_SIZE as u64) as usize;
        let desc = path.to_string_lossy().to_string();
        
        if file_size <= 50 * 1024 * 1024 {
            let mut data = vec![0u8; file_size as usize];
            file.read_exact(&mut data)?;
            let entries = data.chunks_exact(ENTRY_SIZE)
                .map(|c| BookEntry::from_bytes(c.try_into().unwrap())).collect();
            Ok(Self { data: BookData::Memory(entries), entry_count, desc })
        } else {
            Ok(Self { data: BookData::File { path: desc.clone() }, entry_count, desc })
        }
    }

    pub fn probe(&self, board: &Board) -> Vec<BookEntry> {
        self.find_entries(polyglot_hash(board))
    }

    pub fn probe_move(&self, board: &Board) -> Option<Move> {
        let entries = self.probe(board);
        if entries.is_empty() { return None; }
        let total: u32 = entries.iter().map(|e| e.weight as u32).sum();
        if total == 0 { return entries[0].to_chess_move(board); }
        
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(12345);
        let random = seed.wrapping_mul(6364136223846793005).wrapping_add(1) % total as u64;
        
        let mut cum = 0u64;
        for e in &entries {
            cum += e.weight as u64;
            if random < cum { return e.to_chess_move(board); }
        }
        entries[0].to_chess_move(board)
    }

    pub fn probe_best_move(&self, board: &Board) -> Option<Move> {
        self.probe(board).iter().max_by_key(|e| e.weight).and_then(|e| e.to_chess_move(board))
    }

    fn find_entries(&self, key: u64) -> Vec<BookEntry> {
        match &self.data {
            BookData::Memory(e) => self.find_mem(e, key),
            BookData::File { path } => self.find_file(path, key).unwrap_or_default(),
        }
    }

    fn find_mem(&self, entries: &[BookEntry], key: u64) -> Vec<BookEntry> {
        let idx = match entries.binary_search_by_key(&key, |e| e.key) {
            Ok(i) => i, Err(_) => return vec![],
        };
        let mut start = idx;
        while start > 0 && entries[start - 1].key == key { start -= 1; }
        let mut res = vec![];
        let mut i = start;
        while i < entries.len() && entries[i].key == key { res.push(entries[i]); i += 1; }
        res
    }

    fn find_file(&self, path: &str, key: u64) -> io::Result<Vec<BookEntry>> {
        let mut file = FsFile::open(path)?;
        let (mut lo, mut hi) = (0, self.entry_count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            file.seek(SeekFrom::Start((mid * ENTRY_SIZE) as u64))?;
            let mut b = [0u8; 16]; file.read_exact(&mut b)?;
            if BookEntry::from_bytes(&b).key < key { lo = mid + 1; } else { hi = mid; }
        }
        let mut res = vec![]; let mut pos = lo;
        while pos < self.entry_count {
            file.seek(SeekFrom::Start((pos * ENTRY_SIZE) as u64))?;
            let mut b = [0u8; 16]; file.read_exact(&mut b)?;
            let e = BookEntry::from_bytes(&b);
            if e.key != key { break; }
            res.push(e); pos += 1;
        }
        Ok(res)
    }

    pub fn len(&self) -> usize { self.entry_count }
    pub fn is_empty(&self) -> bool { self.entry_count == 0 }
}
