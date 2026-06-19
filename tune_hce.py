import subprocess
import random
import json
import os
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ENGINE_PATH = "./target/release/porcupine"
GAMES_PER_ITERATION = 40   # Total games per iteration
THREADS = 8              # Concurrency in cutechess-cli
NODE_LIMIT = 10000       # Fixed nodes for stable tuning
OUTPUT_FILE = "tuned_hce_params.json"
ITERATIONS = 50

# SPSA parameters
A = 0.1
c = 0.1
alpha = 0.602
gamma = 0.101

# ==============================================================================
# PARAMETER DEFINITIONS
# ==============================================================================
PARAMS = [
    "pawn_mg", "pawn_eg",
    "knight_mg", "knight_eg",
    "bishop_mg", "bishop_eg",
    "rook_mg", "rook_eg",
    "queen_mg", "queen_eg",
    "bishoppair_mg", "bishoppair_eg",
    "passed_rank2_mg", "passed_rank2_eg",
    "passed_rank3_mg", "passed_rank3_eg",
    "passed_rank4_mg", "passed_rank4_eg",
    "passed_rank5_mg", "passed_rank5_eg",
    "passed_rank6_mg", "passed_rank6_eg",
    "passed_rank7_mg", "passed_rank7_eg",
]

INITIAL_THETA = {
    "pawn_mg": 100, "pawn_eg": 120,
    "knight_mg": 320, "knight_eg": 300,
    "bishop_mg": 330, "bishop_eg": 320,
    "rook_mg": 500, "rook_eg": 550,
    "queen_mg": 950, "queen_eg": 1000,
    "bishoppair_mg": 35, "bishoppair_eg": 50,
    "passed_rank2_mg": 5, "passed_rank2_eg": 10,
    "passed_rank3_mg": 10, "passed_rank3_eg": 20,
    "passed_rank4_mg": 20, "passed_rank4_eg": 40,
    "passed_rank5_mg": 40, "passed_rank5_eg": 70,
    "passed_rank6_mg": 70, "passed_rank6_eg": 120,
    "passed_rank7_mg": 120, "passed_rank7_eg": 200,
}

# ==============================================================================
# CUTECHESS-CLI INTERFACE
# ==============================================================================
def run_match(params_plus, params_minus, games, threads):
    def build_engine_args(p_dict, name):
        # Force EvalMethod=Hce
        args = [f"name={name}", f"cmd={ENGINE_PATH}", "proto=uci", "option.EvalMethod=Hce"]
        for k, v in p_dict.items():
            args.append(f"option.{k}={int(v)}")
        return args

    cmd = [
        "cutechess-cli",
        "-engine"
    ] + build_engine_args(params_plus, "Plus") + [
        "-engine"
    ] + build_engine_args(params_minus, "Minus") + [
        "-each", "proto=uci", "tc=inf", f"nodes={NODE_LIMIT}",
        "-games", str(games),
        "-repeat",
        "-concurrency", str(threads),
        "-recover",
        "-openings", "file=openings.epd", "format=epd", "order=random"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    score_plus = 0.5
    found_score = False
    output_log = []
    
    for line in process.stdout:
        output_log.append(line)
        if "Score of Plus vs Minus" in line:
            match = re.search(r"\[(\d+\.\d+)\]", line)
            if match:
                score_plus = float(match.group(1))
                found_score = True
            
        if "Finished game" in line:
            print(".", end="", flush=True)

    process.wait()
    if not found_score:
        print("\nWarning: Could not find score in cutechess-cli output.")
        print("".join(output_log[:20])) # Show just a bit of log
    return score_plus

# ==============================================================================
# SPSA OPTIMIZATION
# ==============================================================================
def optimize():
    theta = INITIAL_THETA.copy()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            theta.update(json.load(f))
        print(f"Loaded existing parameters from {OUTPUT_FILE}", flush=True)

    print(f"Starting SPSA Tuner using cutechess-cli", flush=True)
    
    for k in range(1, ITERATIONS + 1):
        ak = A / (k + 1 + A * 0.1)**alpha
        ck = c / k**gamma
        
        delta = {p: random.choice([-1, 1]) for p in PARAMS}
        
        theta_plus = {p: theta[p] + ck * delta[p] for p in PARAMS}
        theta_minus = {p: theta[p] - ck * delta[p] for p in PARAMS}
        
        print(f"\n--- Iteration {k}/{ITERATIONS} ---", flush=True)
        win_rate_plus = run_match(theta_plus, theta_minus, GAMES_PER_ITERATION, THREADS)
        
        diff = win_rate_plus - 0.5
        print(f"\nIteration {k} Win Rate (Plus): {win_rate_plus:.3f} (Diff: {diff:+.3f})", flush=True)
        
        for p in PARAMS:
            g = diff / (2 * ck * delta[p])
            theta[p] = theta[p] + ak * g
            theta[p] = round(theta[p])
            
        with open(OUTPUT_FILE, "w") as f:
            json.dump(theta, f, indent=4)
            
    print(f"\nTuning complete. Best parameters saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    optimize()
