import numpy as np
import chess
import os
import multiprocessing as mp
from tqdm import tqdm

NUM_KING_BUCKETS = 32
NUM_PIECE_TYPES  = 11
NUM_SQUARES      = 64
SCALE            = 400.0

def _needs_mirror(sq): return (sq % 8) >= 4
def _mirror_h(sq): return sq ^ 7
def _flip_v(sq): return sq ^ 56
def _king_bucket(norm_sq): return (norm_sq // 8) * 4 + (norm_sq % 8)

def compute_halfkp_features(board):
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)

    def perspective(color):
        king_sq = w_king if color == chess.WHITE else b_king
        if color == chess.BLACK: king_sq = _flip_v(king_sq)
        do_mirror = _needs_mirror(king_sq)
        if do_mirror: king_sq = _mirror_h(king_sq)
        bucket = _king_bucket(king_sq)

        indices = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None: continue
            if piece.piece_type == chess.KING and piece.color == color: continue
            
            norm_sq = sq
            if color == chess.BLACK: norm_sq = _flip_v(norm_sq)
            if do_mirror: norm_sq = _mirror_h(norm_sq)
            
            is_friendly = (piece.color == color)
            if is_friendly: pt = piece.piece_type - 1
            else: pt = piece.piece_type - 1 + 5
            
            idx = bucket * (NUM_PIECE_TYPES * NUM_SQUARES) + pt * NUM_SQUARES + norm_sq
            indices.append(idx)
        return indices

    return perspective(chess.WHITE), perspective(chess.BLACK)

record_dtype = np.dtype([
    ('target', np.float32),
    ('num_stm', np.uint8),
    ('num_nstm', np.uint8),
    ('stm_indices', np.int16, (32,)),
    ('nstm_indices', np.int16, (32,)),
])

def process_chunk(lines):
    chunk_out = np.zeros(len(lines), dtype=record_dtype)
    valid_count = 0
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 3: continue
        fen = parts[0]
        score = float(parts[1])
        wdl = float(parts[2])
        
        try: board = chess.Board(fen)
        except: continue
        
        if board.turn == chess.BLACK:
            wdl = 1.0 - wdl
            
        # WDL HOTFIX
        if wdl != 0.5:
            if score > 0: wdl = 1.0
            elif score < 0: wdl = 0.0
            
        w_feat, b_feat = compute_halfkp_features(board)
        if not w_feat or not b_feat: continue
        
        if board.turn == chess.WHITE:
            stm_feat, nstm_feat = w_feat, b_feat
        else:
            stm_feat, nstm_feat = b_feat, w_feat
            
        target = 0.5 * (1.0 / (1.0 + 10.0 ** (-score / SCALE))) + 0.5 * wdl
        
        chunk_out[valid_count]['target'] = target
        chunk_out[valid_count]['num_stm'] = len(stm_feat)
        chunk_out[valid_count]['num_nstm'] = len(nstm_feat)
        
        chunk_out[valid_count]['stm_indices'][:len(stm_feat)] = stm_feat
        chunk_out[valid_count]['nstm_indices'][:len(nstm_feat)] = nstm_feat
        
        valid_count += 1
        
    return chunk_out[:valid_count]

def read_chunks(file_path, chunk_size):
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    input_file = "dataset.txt"
    output_file = "dataset.bin"
    chunk_size = 20000
    
    # Get total file size for progress bar estimation (fen+scores usually ~60 bytes)
    file_size = os.path.getsize(input_file)
    approx_lines = file_size // 60
    
    total_written = 0
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting multiprocessing with {num_workers} workers...")
    
    import itertools
    
    with open(output_file, 'wb') as f_out:
        with mp.Pool(num_workers) as pool:
            chunks = read_chunks(input_file, chunk_size)
            
            # To avoid RAM crashes from greedy iterator consumption,
            # we pull a limited number of chunks at a time.
            max_queued = num_workers * 2
            
            with tqdm(total=approx_lines // chunk_size, desc="Compiling binary dataset") as pbar:
                while True:
                    batch = list(itertools.islice(chunks, max_queued))
                    if not batch:
                        break
                    
                    for out_array in pool.imap(process_chunk, batch):
                        f_out.write(out_array.tobytes())
                        total_written += len(out_array)
                        pbar.update(1)
                
    print(f"Done! Compiled {total_written} positions into {output_file}.")

if __name__ == "__main__":
    main()
