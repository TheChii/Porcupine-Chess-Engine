import os
import shutil

dst_file = "dataset.txt"
merge_file = "dataset_to_merge.txt"
num_to_remove = 130000

# First, we need to count the exact number of lines in dataset.txt to accurately drop the last 130k
print(f"Counting lines in {dst_file}...")
total_lines_dst = 0
with open(dst_file, 'rb') as f:
    for _ in f:
        total_lines_dst += 1

lines_to_keep = total_lines_dst - num_to_remove

print(f"Removing the last {num_to_remove} lines from {dst_file} (keeping {lines_to_keep} lines)...")
with open(dst_file, 'r') as f_in, open('temp_dataset.txt', 'w') as f_out:
    for i in range(lines_to_keep):
        line = f_in.readline()
        if not line:
            break
        f_out.write(line)

os.replace('temp_dataset.txt', dst_file)

print(f"Appending current contents of {merge_file} to {dst_file}...")
# NOTE: dataset_to_merge.txt ALREADY had its first 130k lines removed in the previous operation.
# It currently contains exactly "the rest" of the lines. So we just append all of it.
with open(dst_file, 'a') as f_dst, open(merge_file, 'r') as f_ext:
    shutil.copyfileobj(f_ext, f_dst)

# Optionally, clear dataset_to_merge.txt or leave it as is since it has been fully merged
# open(merge_file, 'w').close() # Uncomment if we want to empty it

print("Fix completed successfully!")
