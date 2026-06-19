import subprocess
import random
import re

ENGINE_PATH = "./target/release/porcupine"

def build_args(p_dict, name):
    args = [f"-engine name={name}", f"cmd={ENGINE_PATH}", "option.EvalMethod=Hce"]
    for k, v in p_dict.items():
        args.append(f"option.{k}={int(v)}")
    return " ".join(args)

# Test with default values
params = {
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

cmd = [
    "cutechess-cli",
    build_args(params, "Plus"),
    build_args(params, "Minus"),
    "-each", "proto=uci", "tc=inf", "nodes=1000",
    "-games", "4",
    "-repeat",
    "-concurrency", "1"
]

full_cmd = " ".join(cmd)
print(f"RUNNING DEBUG COMMAND:\n{full_cmd}\n")

process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in process.stdout:
    print(line, end="")

process.wait()
print(f"\nExit Code: {process.returncode}")
