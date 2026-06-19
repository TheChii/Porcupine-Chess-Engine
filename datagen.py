import multiprocessing
import subprocess
import random
import signal
import sys
import time
import os
import atexit
import io
import chess
import chess.pgn
import queue
import threading

# === Configuration ===
ENGINE_PATH = "./target/release/porcupine"
BOOKS = {
    # 1. Core strength backbone — 80%
    "books/UHO_4060_v4.epd": {"weight": 20, "type": "epd", "skip_plies": 4},
    "books/UHO_XXL_2022_+120_+149.pgn": {"weight": 20, "type": "pgn", "skip_plies": 6},
    "books/UHO_MEGA_2022_+110_+149.pgn": {"weight": 60, "type": "pgn", "skip_plies": 6},

    # 2. Human + engine realism blend — 5%
    "books/popularpos_lichess_v3.epd": {"weight": 0, "type": "epd", "skip_plies": 4},
    "books/bjbraams_chessdb_198350_lines.pgn": {"weight": 0, "type": "pgn", "skip_plies": 6},

    # 3. Sharp / high-pressure positions — 10%
    "books/UHO_Lichess_4852_v1.epd": {"weight": 0, "type": "epd", "skip_plies": 4},
    "books/UHO_XXL_+1.00_+1.29.pgn": {"weight": 0, "type": "pgn", "skip_plies": 6},
    "books/8mvs_big_+80_+109.epd": {"weight": 0, "type": "epd", "skip_plies": 4},

    # 4. Opening diversity layer — 5%
    "books/2moves_v2.pgn": {"weight": 0, "type": "pgn", "skip_plies": 4},
    "books/4mvs_+90_+99.epd": {"weight": 0, "type": "epd", "skip_plies": 4}
}
OUTPUT_FILE = "dataset.txt"
DEPTH = 12
SAVE_INTERVAL = 1  
MAX_MOVES = 70     
ADJUDICATION_THRESHOLD = 500 
# =====================

class Engine:
    def __init__(self, path):
        self.process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._enqueue_output)
        self.t.daemon = True
        self.t.start()
        
        self.send("uci")
        while self.readline() != "uciok": pass
        self.send("isready")
        while self.readline() != "readyok": pass

    def _enqueue_output(self):
        for line in iter(self.process.stdout.readline, ''):
            self.q.put(line)
        self.process.stdout.close()

    def send(self, cmd):
        if self.process.poll() is None:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def readline(self, timeout=None):
        try:
            line = self.q.get(timeout=timeout)
            return line.strip()
        except queue.Empty:
            if self.process.poll() is not None:
                return ""
            return None

    def get_move_and_score(self, fen, depth):
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        
        bestmove, score_cp, is_mate, mate_in = None, 0, False, 0
        start_time = time.time()
        timeout_seconds = 60
        
        while True:
            line = self.readline(timeout=1.0)
            if line is None:
                if time.time() - start_time > timeout_seconds:
                    print(f"Engine timeout on fen: {fen}")
                    break
                continue
            if not line: break
            if line.startswith("info"):
                parts = line.split()
                if "score" in parts:
                    idx = parts.index("score")
                    try:
                        score_type, score_val = parts[idx+1], parts[idx+2]
                        if score_type == "cp":
                            score_cp = int(score_val)
                            is_mate = False
                        elif score_type == "mate":
                            is_mate = True
                            clean_val = "".join(c for c in score_val if c.isdigit() or c == '-')
                            mate_in = int(clean_val)
                    except (ValueError, IndexError): pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1: bestmove = parts[1]
                break
        return bestmove, score_cp, is_mate, mate_in

    def quit(self):
        try:
            self.send("quit")
            self.process.wait(timeout=1)
        except:
            if self.process.poll() is None: self.process.kill()

def parse_fen_plies(fen):
    try:
        parts = fen.split()
        plies = (int(parts[5]) - 1) * 2
        if parts[1] == 'b': plies += 1
        return plies
    except: return 0

def get_side_to_move(fen, moves):
    initial_turn = fen.split()[1]
    return initial_turn if len(moves) % 2 == 0 else ('b' if initial_turn == 'w' else 'w')

def normalize_fen(fen):
    return " ".join(fen.split()[:4])

def load_books(books_config):
    loaded_books = {}
    for f, config in books_config.items():
        if os.path.exists(f):
            print(f"Loading {f}...")
            items = []
            if config["type"] == "epd":
                with open(f, 'r') as fd:
                    for line in fd:
                        if line.strip(): items.append(line.strip().split(';')[0].strip())
            elif config["type"] == "pgn":
                with open(f, 'r') as fd:
                    current_moves = []
                    for line in fd:
                        line = line.strip()
                        if not line or line.startswith('['):
                            continue
                        current_moves.append(line)
                        if '1/2-1/2' in line or '1-0' in line or '0-1' in line or '*' in line:
                            items.append(" ".join(current_moves))
                            current_moves = []
            if items:
                loaded_books[f] = items
            else:
                print(f"Warning: No valid lines found in {f}")
        else:
            print(f"Warning: {f} not found!")
    return loaded_books

def load_existing_fens(file):
    fens = set()
    if os.path.exists(file):
        print(f"Loading existing positions from {file}...")
        with open(file, 'r') as f:
            for line in f:
                parts = line.split('|')
                if parts: fens.add(hash(normalize_fen(parts[0])))
    return fens

def worker_process(worker_id, loaded_books, write_queue, stop_event):
    # Ignore SIGINT in the worker so the main process handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    random.seed()
    
    engine = Engine(ENGINE_PATH)
    
    def cleanup():
        engine.quit()
    atexit.register(cleanup)
    
    book_keys = list(loaded_books.keys())
    book_weights = [BOOKS[k]["weight"] for k in book_keys]

    def get_pgn_board(moves_str):
        pgn = io.StringIO(moves_str)
        game = chess.pgn.read_game(pgn)
        if game:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            return board
        return chess.Board(chess960=True)

    games_played = 0
    while not stop_event.is_set():
        book_key = random.choices(book_keys, weights=book_weights)[0]
        book_type = BOOKS[book_key]["type"]
        skip_plies = BOOKS[book_key]["skip_plies"]
        book_data = loaded_books[book_key]
        
        opening_data = random.choice(book_data)
        if book_type == "epd":
            board = chess.Board(opening_data, chess960=True)
        elif book_type == "pgn":
            board = get_pgn_board(opening_data)
            
        opening = board.fen()
        game_entries = []
        initial_plies = parse_fen_plies(opening)
        result, save_offset = 0.5, random.randint(0, SAVE_INTERVAL - 1)
        
        # 30% chance to play 1-2 random moves to introduce realistic blunders/imbalances
        # without completely destroying the positional structure
        random_plies = random.randint(1, 2) if random.random() < 0.3 else 0
        
        engine.send("ucinewgame")
        engine.send("isready")
        while engine.readline() != "readyok": pass
        
        plies_played = 0
        adj_counter = 0
        while plies_played < MAX_MOVES * 2:
            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                if outcome is not None:
                    if outcome.winner is None:
                        result = 0.5
                    elif outcome.winner == chess.WHITE:
                        result = 1.0
                    else:
                        result = 0.0
                break

            current_ply = initial_plies + plies_played
            current_fen = board.fen()
            
            if plies_played < random_plies:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                bestmove = random.choice(legal_moves).uci()
                # Skip recording score for random moves, just make the move
                score_cp, is_mate, mate_in = 0, False, 0
            else:
                bestmove, score_cp, is_mate, mate_in = engine.get_move_and_score(current_fen, DEPTH)
            
            if bestmove in [None, "0000", "(none)"]:
                if is_mate:
                    side = 'w' if board.turn else 'b'
                    result = (1.0 if side == 'w' else 0.0) if mate_in > 0 else (1.0 if side == 'b' else 0.0)
                break
            
            if current_ply >= skip_plies and plies_played >= random_plies and (plies_played % SAVE_INTERVAL == save_offset):
                game_entries.append((current_fen, score_cp, bestmove))

            if is_mate:
                side = 'w' if board.turn else 'b'
                result = (1.0 if side == 'w' else 0.0) if mate_in > 0 else (1.0 if side == 'b' else 0.0)
                break

            try:
                board.push_uci(bestmove)
            except ValueError:
                break
            
            plies_played += 1

            if abs(score_cp) > ADJUDICATION_THRESHOLD:
                adj_counter += 1
                if adj_counter >= 6:
                    # score_cp was from pre-push STM; board.turn has flipped
                    pre_push_stm = 'b' if board.turn else 'w'
                    result = (1.0 if pre_push_stm == 'w' else 0.0) if score_cp > 0 else (1.0 if pre_push_stm == 'b' else 0.0)
                    break
            else:
                adj_counter = 0

        # Put results onto the queue
        if game_entries:
            write_queue.put((game_entries, result))

def main():
    loaded_books = load_books(BOOKS)
    if not loaded_books: return
    existing_fens = load_existing_fens(OUTPUT_FILE)
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    user_input_cores = input("Number of cores: ")
    if user_input_cores:
        num_cores = int(user_input_cores)

    print(f"Starting {num_cores} worker processes (Depth {DEPTH}). Press Ctrl-C to stop.")

    write_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    
    # Start worker processes
    workers = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=worker_process, args=(i, loaded_books, write_queue, stop_event))
        p.start()
        workers.append(p)
    
    games_count = 0
    
    buffer_lines = []

    def signal_handler(sig, frame):
        if stop_event.is_set():
            print("\nForce quitting...")
            for p in workers:
                p.terminate()
            sys.exit(1)
        print("\nStop signal received! Waiting for workers to finish their current games... (Press Ctrl-C again to force quit)")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)

    total_positions = len(existing_fens)
    last_100_time = time.time()
    times_per_100_games = []

    try:
        with open(OUTPUT_FILE, 'a') as f:
            while True:
                try:
                    game_entries, result = write_queue.get(timeout=1.0)
                except queue.Empty:
                    if stop_event.is_set() and not any(p.is_alive() for p in workers):
                        break
                    continue
                
                new_positions = 0
                # Write batched data to memory buffer
                for fen, score, best_move in game_entries:
                    norm_hash = hash(normalize_fen(fen))
                    if norm_hash not in existing_fens:
                        buffer_lines.append(f"{fen}|{score}|{result}|{best_move}|{DEPTH}\n")
                        existing_fens.add(norm_hash)
                        new_positions += 1
                
                total_positions += new_positions
                games_count += 1
                
                if games_count % 100 == 0:
                    elapsed = time.time() - last_100_time
                    times_per_100_games.append(elapsed)
                    sorted_times = sorted(times_per_100_games)
                    median_time = sorted_times[len(sorted_times) // 2]
                    median_speed = 100 / median_time if median_time > 0 else 0
                    
                    if games_count % 200 == 0:
                        if buffer_lines:
                            f.writelines(buffer_lines)
                            f.flush()
                            buffer_lines.clear()
                        print(f"Games played: {games_count} | Unique pos: {total_positions} | Median speed: {median_speed:.1f} games/s (Saved to file)")
                    else:
                        print(f"Games played: {games_count} | Unique pos: {total_positions} | Median speed: {median_speed:.1f} games/s")
                    
                    last_100_time = time.time()

            if buffer_lines:
                print(f"Saving {len(buffer_lines)} remaining positions to file...")
                f.writelines(buffer_lines)
                f.flush()
                buffer_lines.clear()
            print("Done. All workers exited cleanly.")
            
    except KeyboardInterrupt:
        pass # Handled by signal handler

if __name__ == "__main__":
    main()

