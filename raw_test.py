import subprocess
import os

# Exact parameters from the tuner
ENGINE_PATH = "./target/release/porcupine"
PARAMS = "option.EvalMethod=Hce option.pawn_mg=100 option.pawn_eg=120" # Shortened for test

cmd = [
    "cutechess-cli",
    f"-engine name=Plus cmd={ENGINE_PATH} {PARAMS}",
    f"-engine name=Minus cmd={ENGINE_PATH} {PARAMS}",
    "-each", "proto=uci", "tc=inf", "nodes=1000",
    "-games", "2",
    "-repeat",
    "-concurrency", "1"
]

full_cmd = " ".join(cmd)
print(f"Executing: {full_cmd}\n")

process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

print("--- RAW OUTPUT START ---")
for line in process.stdout:
    print(repr(line)) # Print with repr to see hidden characters
print("--- RAW OUTPUT END ---")

process.wait()
print(f"\nExit Code: {process.returncode}")
