//! Polyglot opening book format reader.

use super::zobrist::polyglot_hash;
use crate::types::{Board, Move, Piece};
use movegen::{File, Rank, Square};
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
    fn from_bytes(b: &[u8; 16]) -> Self {
        Self {
            key: u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            raw_move: u16::from_be_bytes([b[8], b[9]]),
            weight: u16::from_be_bytes([b[10], b[11]]),
            learn: u32::from_be_bytes([b[12], b[13], b[14], b[15]]),
        }
    }

    pub fn decode_move(&self) -> (Square, Square, Option<Piece>) {
        let tf = (self.raw_move & 0x7) as u8;
        let tr = ((self.raw_move >> 3) & 0x7) as u8;
        let ff = ((self.raw_move >> 6) & 0x7) as u8;
        let fr = ((self.raw_move >> 9) & 0x7) as u8;
        let p = ((self.raw_move >> 12) & 0x7) as usize;

        let from = Square::from_file_rank(File::from_index(ff).unwrap(), Rank::from_index(fr).unwrap());
        let to = Square::from_file_rank(File::from_index(tf).unwrap(), Rank::from_index(tr).unwrap());

        let promo = match p { 1 => Some(Piece::Knight), 2 => Some(Piece::Bishop), 3 => Some(Piece::Rook), 4 => Some(Piece::Queen), _ => None };
        (from, to, promo)
    }

    pub fn to_chess_move(&self, b: &Board) -> Option<Move> {
        let (f, t, p) = self.decode_move();
        let at = self.adjust_castling_move(b, f, t);
        for m in b.generate_moves().iter() {
            if m.from() == f && m.to() == at {
                if p.is_some() { if m.flag().promotion_piece() == p { return Some(m); } }
                else if m.flag().promotion_piece().is_none() { return Some(m); }
            }
        }
        None
    }

    fn adjust_castling_move(&self, b: &Board, f: Square, t: Square) -> Square {
        if let Some((p, _)) = b.piece_at(f) {
            if p == Piece::King && f.file() == File::E {
                if t.file() == File::H { return Square::from_file_rank(File::G, t.rank()); }
                if t.file() == File::A { return Square::from_file_rank(File::C, t.rank()); }
            }
        }
        t
    }
}

pub struct PolyglotBook {
    data: BookData,
    entry_count: usize,
    pub desc: String,
}

enum BookData {
    Memory(Vec<BookEntry>),
    File { path: String },
}

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
            let entries = data
                .chunks_exact(ENTRY_SIZE)
                .map(|c| BookEntry::from_bytes(c.try_into().unwrap()))
                .collect();
            Ok(Self {
                data: BookData::Memory(entries),
                entry_count,
                desc,
            })
        } else {
            Ok(Self {
                data: BookData::File { path: desc.clone() },
                entry_count,
                desc,
            })
        }
    }

    pub fn probe(&self, b: &Board) -> Vec<BookEntry> { self.find_entries(polyglot_hash(b)) }

    pub fn probe_move(&self, b: &Board) -> Option<Move> {
        let es = self.probe(b);
        if es.is_empty() { return None; }
        let t: u32 = es.iter().map(|e| e.weight as u32).sum();
        if t == 0 { return es[0].to_chess_move(b); }
        let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(12345);
        let r = s.wrapping_mul(6364136223846793005).wrapping_add(1) % t as u64;
        let mut c = 0u64;
        for e in &es {
            c += e.weight as u64;
            if r < c { return e.to_chess_move(b); }
        }
        es[0].to_chess_move(b)
    }

    pub fn probe_best_move(&self, b: &Board) -> Option<Move> {
        self.probe(b).iter().max_by_key(|e| e.weight).and_then(|e| e.to_chess_move(b))
    }

    fn find_entries(&self, k: u64) -> Vec<BookEntry> {
        match &self.data {
            BookData::Memory(e) => self.find_mem(e, k),
            BookData::File { path } => self.find_file(path, k).unwrap_or_default(),
        }
    }

    fn find_mem(&self, es: &[BookEntry], k: u64) -> Vec<BookEntry> {
        let i = match es.binary_search_by_key(&k, |e| e.key) { Ok(i) => i, Err(_) => return vec![] };
        let mut s = i;
        while s > 0 && es[s - 1].key == k { s -= 1; }
        let mut r = vec![];
        let mut j = s;
        while j < es.len() && es[j].key == k { r.push(es[j]); j += 1; }
        r
    }

    fn find_file(&self, path: &str, key: u64) -> io::Result<Vec<BookEntry>> {
        let mut file = FsFile::open(path)?;
        let (mut lo, mut hi) = (0, self.entry_count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            file.seek(SeekFrom::Start((mid * ENTRY_SIZE) as u64))?;
            let mut b = [0u8; 16];
            file.read_exact(&mut b)?;
            if BookEntry::from_bytes(&b).key < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut res = vec![];
        let mut pos = lo;
        while pos < self.entry_count {
            file.seek(SeekFrom::Start((pos * ENTRY_SIZE) as u64))?;
            let mut b = [0u8; 16];
            file.read_exact(&mut b)?;
            let e = BookEntry::from_bytes(&b);
            if e.key != key {
                break;
            }
            res.push(e);
            pos += 1;
        }
        Ok(res)
    }

    pub fn len(&self) -> usize { self.entry_count }
    pub fn is_empty(&self) -> bool { self.entry_count == 0 }
}
