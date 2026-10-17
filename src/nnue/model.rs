//! Versioned binary NNUE model format loader and validator.
//!
//! # Binary File Specification (`.nnue`)
//!
//! All integer fields are little-endian.
//!
//! ## Header (44 bytes)
//! - `magic`: [u8; 4] = b"FNUE" (0x45554E46)
//! - `format_version`: u32 (1)
//! - `feature_version`: u32 (1 for HalfKAv2 canonical 22,528 features)
//! - `arch_id`: u32 (1 for 1024-16-32-1)
//! - `has_psqt`: u32 (0 = disabled, 1 = enabled)
//! - `num_features`: u32 (22528)
//! - `transformer_width`: u32 (512)
//! - `dense1_out`: u32 (16)
//! - `dense2_out`: u32 (32)
//! - `psqt_channels`: u32 (8)
//! - `checksum`: u32 (Adler-32 of all payload bytes)
//!
//! ## Payload
//! 1. `feature_weights`: [22528 * 512] i16
//! 2. `feature_biases`: [512] i16
//! 3. If `has_psqt == 1`:
//!    - `psqt_weights`: [22528 * 8] i32
//!    - `psqt_biases`: [8] i32
//!    - `psqt_head`: [8] i32
//! 4. `dense1_weights`: [16 * 1024] i8
//! 5. `dense1_biases`: [16] i32
//! 6. `dense2_weights`: [32 * 16] i8
//! 7. `dense2_biases`: [32] i32
//! 8. `output_weights`: [32] i8
//! 9. `output_bias`: [1] i32

use super::accumulator::{PSQT_CHANNELS, TRANSFORMER_WIDTH};
use super::features::NUM_FEATURES;
use super::network::{NetworkHead, DENSE1_IN, DENSE1_OUT, DENSE2_OUT};
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub const MAGIC: [u8; 4] = *b"FNUE";
pub const FORMAT_VERSION: u32 = 1;
pub const FEATURE_VERSION: u32 = 1;
pub const ARCH_ID: u32 = 1;

/// Errors that can occur when loading an NNUE model.
#[derive(Debug)]
pub enum ModelError {
    Io(io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedFormatVersion(u32),
    UnsupportedFeatureVersion(u32),
    UnsupportedArchitecture(u32),
    DimensionMismatch { expected: usize, found: usize },
    ChecksumMismatch { expected: u32, calculated: u32 },
    TruncatedFile { expected_bytes: usize, found_bytes: usize },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "IO error: {}", e),
            ModelError::InvalidMagic(m) => write!(f, "Invalid magic bytes: {:?}", m),
            ModelError::UnsupportedFormatVersion(v) => write!(f, "Unsupported format version: {}", v),
            ModelError::UnsupportedFeatureVersion(v) => write!(f, "Unsupported feature version: {}", v),
            ModelError::UnsupportedArchitecture(a) => write!(f, "Unsupported architecture: {}", a),
            ModelError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, found {}", expected, found)
            }
            ModelError::ChecksumMismatch { expected, calculated } => {
                write!(
                    f,
                    "Checksum mismatch: expected 0x{:08X}, calculated 0x{:08X}",
                    expected, calculated
                )
            }
            ModelError::TruncatedFile { expected_bytes, found_bytes } => {
                write!(
                    f,
                    "Truncated file: expected {} bytes, found {}",
                    expected_bytes, found_bytes
                )
            }
        }
    }
}

impl std::error::Error for ModelError {}

impl From<io::Error> for ModelError {
    fn from(e: io::Error) -> Self {
        ModelError::Io(e)
    }
}

/// Calculate Adler-32 checksum.
pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    const MOD_ADLER: u32 = 65521;

    for &byte in data {
        a = (a + byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    (b << 16) | a
}

/// Complete NNUE model loaded in memory.
#[derive(Clone)]
pub struct NnueModel {
    /// Transformer weights: [22528][512]
    pub feature_weights: Vec<[i16; TRANSFORMER_WIDTH]>,
    /// Transformer biases: [512]
    pub feature_biases: [i16; TRANSFORMER_WIDTH],
    /// PSQT weights: [22528][8]
    pub psqt_weights: Option<Vec<[i32; PSQT_CHANNELS]>>,
    /// PSQT biases: [8]
    pub psqt_biases: [i32; PSQT_CHANNELS],
    /// Dense neural network head
    pub head: NetworkHead,
}

impl NnueModel {
    /// Create a zeroed baseline model (useful as fallback and testing).
    pub fn create_default() -> Self {
        let mut feature_weights = Vec::with_capacity(NUM_FEATURES);
        for _ in 0..NUM_FEATURES {
            feature_weights.push([0i16; TRANSFORMER_WIDTH]);
        }

        Self {
            feature_weights,
            feature_biases: [0i16; TRANSFORMER_WIDTH],
            psqt_weights: None,
            psqt_biases: [0i32; PSQT_CHANNELS],
            head: NetworkHead::default(),
        }
    }

    /// Load model from a file path.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Self::from_bytes(&buffer)
    }

    /// Load and validate model from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        const HEADER_SIZE: usize = 44;
        if bytes.len() < HEADER_SIZE {
            return Err(ModelError::TruncatedFile {
                expected_bytes: HEADER_SIZE,
                found_bytes: bytes.len(),
            });
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(ModelError::InvalidMagic(magic));
        }

        let format_version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if format_version != FORMAT_VERSION {
            return Err(ModelError::UnsupportedFormatVersion(format_version));
        }

        let feature_version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if feature_version != FEATURE_VERSION {
            return Err(ModelError::UnsupportedFeatureVersion(feature_version));
        }

        let arch_id = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if arch_id != ARCH_ID {
            return Err(ModelError::UnsupportedArchitecture(arch_id));
        }

        let has_psqt = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) != 0;
        let num_features = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let transformer_width = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        let dense1_out = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        let dense2_out = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
        let psqt_channels = u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize;
        let expected_checksum = u32::from_le_bytes(bytes[40..44].try_into().unwrap());

        if num_features != NUM_FEATURES {
            return Err(ModelError::DimensionMismatch {
                expected: NUM_FEATURES,
                found: num_features,
            });
        }
        if transformer_width != TRANSFORMER_WIDTH {
            return Err(ModelError::DimensionMismatch {
                expected: TRANSFORMER_WIDTH,
                found: transformer_width,
            });
        }
        if dense1_out != DENSE1_OUT {
            return Err(ModelError::DimensionMismatch {
                expected: DENSE1_OUT,
                found: dense1_out,
            });
        }
        if dense2_out != DENSE2_OUT {
            return Err(ModelError::DimensionMismatch {
                expected: DENSE2_OUT,
                found: dense2_out,
            });
        }
        if psqt_channels != PSQT_CHANNELS {
            return Err(ModelError::DimensionMismatch {
                expected: PSQT_CHANNELS,
                found: psqt_channels,
            });
        }

        // Validate payload length
        let payload = &bytes[HEADER_SIZE..];
        let ft_weights_bytes = NUM_FEATURES * TRANSFORMER_WIDTH * 2;
        let ft_biases_bytes = TRANSFORMER_WIDTH * 2;

        let psqt_bytes = if has_psqt {
            NUM_FEATURES * PSQT_CHANNELS * 4 + PSQT_CHANNELS * 4 + PSQT_CHANNELS * 4
        } else {
            0
        };

        let dense_bytes = (DENSE1_OUT * DENSE1_IN) // d1 weights i8
            + (DENSE1_OUT * 4) // d1 biases i32
            + (DENSE1_OUT) // out weights i8
            + 4; // out bias i32

        let expected_payload_bytes = ft_weights_bytes + ft_biases_bytes + psqt_bytes + dense_bytes;
        if payload.len() != expected_payload_bytes {
            return Err(ModelError::TruncatedFile {
                expected_bytes: HEADER_SIZE + expected_payload_bytes,
                found_bytes: bytes.len(),
            });
        }

        // Verify checksum
        let calculated_checksum = adler32(payload);
        if calculated_checksum != expected_checksum {
            return Err(ModelError::ChecksumMismatch {
                expected: expected_checksum,
                calculated: calculated_checksum,
            });
        }

        let mut offset = 0;

        // 1. Feature transformer weights: [22528][512] i16
        let mut feature_weights = Vec::with_capacity(NUM_FEATURES);
        for _ in 0..NUM_FEATURES {
            let mut row = [0i16; TRANSFORMER_WIDTH];
            for j in 0..TRANSFORMER_WIDTH {
                row[j] = i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap());
                offset += 2;
            }
            feature_weights.push(row);
        }

        // 2. Feature transformer biases: [512] i16
        let mut feature_biases = [0i16; TRANSFORMER_WIDTH];
        for j in 0..TRANSFORMER_WIDTH {
            feature_biases[j] = i16::from_le_bytes(payload[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }

        // 3. PSQT weights and biases (optional)
        let (psqt_weights, psqt_biases, psqt_head_weights) = if has_psqt {
            let mut pw = Vec::with_capacity(NUM_FEATURES);
            for _ in 0..NUM_FEATURES {
                let mut row = [0i32; PSQT_CHANNELS];
                for j in 0..PSQT_CHANNELS {
                    row[j] = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
                    offset += 4;
                }
                pw.push(row);
            }
            let mut pb = [0i32; PSQT_CHANNELS];
            for j in 0..PSQT_CHANNELS {
                pb[j] = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            let mut ph = [0i32; PSQT_CHANNELS];
            for j in 0..PSQT_CHANNELS {
                ph[j] = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            (Some(pw), pb, ph)
        } else {
            (None, [0i32; PSQT_CHANNELS], [0i32; PSQT_CHANNELS])
        };

        // 4. Dense layer 1: [16][1024] i8, biases [16] i32
        let mut d1_weights = [[0i8; DENSE1_IN]; DENSE1_OUT];
        for i in 0..DENSE1_OUT {
            for j in 0..DENSE1_IN {
                d1_weights[i][j] = payload[offset] as i8;
                offset += 1;
            }
        }
        let mut d1_biases = [0i32; DENSE1_OUT];
        for i in 0..DENSE1_OUT {
            d1_biases[i] = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }

        // 5. Output layer: [128] i8, bias [1] i32
        let mut out_weights = [0i8; DENSE1_OUT];
        for i in 0..DENSE1_OUT {
            out_weights[i] = payload[offset] as i8;
            offset += 1;
        }
        let out_bias = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
        offset += 4;

        debug_assert_eq!(offset, expected_payload_bytes);

        let head = NetworkHead {
            d1_weights,
            d1_biases,
            out_weights,
            out_bias,
            psqt_weights: psqt_head_weights,
            has_psqt,
        };

        Ok(Self {
            feature_weights,
            feature_biases,
            psqt_weights,
            psqt_biases,
            head,
        })
    }
}
