import subprocess

def run_uci(commands):
    p = subprocess.Popen(['./target/release/porcupine'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = []
    for cmd in commands:
        p.stdin.write(cmd + "\n")
        p.stdin.flush()
    
    while True:
        line = p.stdout.readline()
        if not line: break
        out.append(line.strip())
        if "bestmove" in line:
            break
    
    p.terminate()
    return out

commands = [
    "uci",
    "isready",
    "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1",
    "go depth 1"
]
for line in run_uci(commands):
    print(line)
