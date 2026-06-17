import multiprocessing
import multiprocessing.pool
import subprocess
import random
import signal
import sys
import time
import os
import atexit

# === Configuration ===
ENGINE_PATH = "./target/release/porcupine"
EPD_FILES = ["noob_4moves.epd", "UHO_Lichess_4852_v1.epd"]
OUTPUT_FILE = "dataset.txt"
DEPTH = 12
SAVE_INTERVAL = 4   # Save 1 position every 4 plies
SKIP_PLIES = 6      # Skip first 6 plies of the game
MAX_MOVES = 200     # Max full moves (400 plies)
ADJUDICATION_THRESHOLD = 1500 # Score in CP to end game early
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
        self.send("uci")
        while self.readline() != "uciok": pass
        self.send("isready")
        while self.readline() != "readyok": pass

    def send(self, cmd):
        if self.process.poll() is None:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def readline(self):
        return "" if self.process.poll() is not None else self.process.stdout.readline().strip()

    def get_move_and_score(self, fen, moves, depth):
        if moves:
            self.send(f"position fen {fen} moves {' '.join(moves)}")
        else:
            self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        
        bestmove, score_cp, is_mate, mate_in = None, 0, False, 0
        while True:
            line = self.readline()
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

    def get_current_fen(self, fen, moves):
        if moves:
            self.send(f"position fen {fen} moves {' '.join(moves)}")
        else:
            self.send(f"position fen {fen}")
        self.send("d")
        current_fen = ""
        while True:
            line = self.readline()
            if not line: break
            if line.startswith("FEN:"): current_fen = line.replace("FEN:", "").strip()
            if line.startswith("Side to move:"): break
        return current_fen

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

def load_openings(files):
    openings = []
    for f in files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            with open(f, 'r') as fd:
                for line in fd:
                    if line.strip(): openings.append(line.strip().split(';')[0].strip())
    return openings

def load_existing_fens(file):
    fens = set()
    if os.path.exists(file):
        print(f"Loading existing positions from {file}...")
        with open(file, 'r') as f:
            for line in f:
                parts = line.split('|')
                if parts: fens.add(normalize_fen(parts[0]))
    return fens

def worker_process(worker_id, openings, write_queue):
    # Ignore SIGINT in the worker so the main process handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    random.seed()
    
    engine = Engine(ENGINE_PATH)
    
    def cleanup():
        engine.quit()
    atexit.register(cleanup)
    
    games_played = 0
    while True:
        opening = random.choice(openings)
        moves, game_entries = [], []
        initial_plies = parse_fen_plies(opening)
        result, save_offset = 0.5, random.randint(0, SAVE_INTERVAL - 1)
        
        while len(moves) < MAX_MOVES * 2:
            current_ply, plies_played = initial_plies + len(moves), len(moves)
            current_fen = engine.get_current_fen(opening, moves)
            if not current_fen: break
            
            bestmove, score_cp, is_mate, mate_in = engine.get_move_and_score(opening, moves, DEPTH)
            
            if bestmove in [None, "0000", "(none)"]:
                if is_mate:
                    side = get_side_to_move(opening, moves)
                    result = (1.0 if side == 'w' else 0.0) if mate_in > 0 else (1.0 if side == 'b' else 0.0)
                break
            
            if current_ply >= SKIP_PLIES and (plies_played % SAVE_INTERVAL == save_offset):
                game_entries.append((current_fen, score_cp, bestmove))

            if is_mate:
                side = get_side_to_move(opening, moves)
                result = (1.0 if side == 'w' else 0.0) if mate_in > 0 else (1.0 if side == 'b' else 0.0)
                break

            moves.append(bestmove)
            if abs(score_cp) > ADJUDICATION_THRESHOLD:
                side = get_side_to_move(opening, moves)
                result = (1.0 if side == 'w' else 0.0) if score_cp > 0 else (1.0 if side == 'b' else 0.0)
                break

        # Put results onto the queue
        if game_entries:
            write_queue.put((game_entries, result))

def main():
    openings = load_openings(EPD_FILES)
    if not openings: return
    existing_fens = load_existing_fens(OUTPUT_FILE)
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting {num_cores} worker processes (Depth {DEPTH}). Press Ctrl-C to stop.")

    manager = multiprocessing.Manager()
    write_queue = manager.Queue()
    
    # Start worker processes
    pool = multiprocessing.Pool(processes=num_cores, initializer=worker_process, initargs=(0, openings, write_queue))
    
    games_count = 0
    
    def signal_handler(sig, frame):
        print("\nStopping gracefully. Terminating workers...")
        pool.terminate()
        pool.join()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            # Wait for data from workers
            game_entries, result = write_queue.get()
            
            new_positions = 0
            # Write batched data to file to prevent corruption
            with open(OUTPUT_FILE, 'a') as f:
                for fen, score, best_move in game_entries:
                    norm = normalize_fen(fen)
                    if norm not in existing_fens:
                        f.write(f"{fen}|{score}|{result}|{best_move}|{DEPTH}\n")
                        existing_fens.add(norm)
                        new_positions += 1
            
            games_count += 1
            if games_count % 10 == 0:
                print(f"Games played: {games_count} | Total unique positions saved: {len(existing_fens)}")

            
    except KeyboardInterrupt:
        pass # Handled by signal handler

if __name__ == "__main__":
    main()

