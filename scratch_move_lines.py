import os
import shutil

src_file = "dataset_to_merge.txt"
dst_file = "dataset.txt"
num_lines = 130000

print(f"Extracting first {num_lines} lines from {src_file}...")
with open(src_file, 'r') as f_in, open('extracted_lines.txt', 'w') as f_out:
    for i in range(num_lines):
        line = f_in.readline()
        if not line:
            break
        f_out.write(line)

print(f"Appending lines to {dst_file}...")
with open(dst_file, 'a') as f_dst, open('extracted_lines.txt', 'r') as f_ext:
    shutil.copyfileobj(f_ext, f_dst)

print(f"Writing remaining lines back to {src_file}...")
with open(src_file, 'r') as f_in, open('temp_remaining.txt', 'w') as f_out:
    # Skip the first num_lines
    for _ in range(num_lines):
        if not f_in.readline():
            break
    # Copy the rest
    shutil.copyfileobj(f_in, f_out)

os.replace('temp_remaining.txt', src_file)
os.remove('extracted_lines.txt')
print("Operation completed successfully!")
