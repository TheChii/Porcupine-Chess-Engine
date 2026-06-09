import subprocess
import time

p = subprocess.Popen(['./target/release/porcupine'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
p.stdin.write("uci\n")
p.stdin.flush()
time.sleep(0.5)
p.stdin.write("go infinite\n")
p.stdin.flush()
time.sleep(2)
p.stdin.write("stop\n")
p.stdin.flush()
time.sleep(2)
p.kill()
for line in p.stdout:
    print(line, end="")
