"""
Process and filter Lichess Chess Position Evaluations dataset.

Features:
- Stream reads row groups from downloaded parquet files to keep RAM < 500 MB.
- Filters by engine search depth threshold.
- Converts White-relative evaluations to Side-to-Move (STM) perspective.
- Caps dataset between 30M and 60M positions as requested.
- Compresses output into a clean, compact parquet format (~1.2 - 1.8 GB).
"""

import os
import sys
import time
import argparse
import pyarrow as pa
import pyarrow.parquet as pq


def analyze_depth_distribution(parquet_path: str, sample_row_groups: int = 5):
    """Inspects the depth distribution across row groups to determine optimal threshold."""
    pf = pq.ParquetFile(parquet_path)
    total_rgs = pf.num_row_groups
    print(f"Total rows in {parquet_path}: {pf.metadata.num_rows:,} across {total_rgs} row groups.")

    depth_counts = {}
    rgs_to_check = min(sample_row_groups, total_rgs)
    sampled_rows = 0

    for i in range(rgs_to_check):
        table = pf.read_row_group(i, columns=["depth"])
        depths = table["depth"].to_pylist()
        sampled_rows += len(depths)
        for d in depths:
            if d is not None:
                bucket = (d // 5) * 5
                depth_counts[bucket] = depth_counts.get(bucket, 0) + 1

    print(f"\nSampled {sampled_rows:,} rows across {rgs_to_check} row groups. Depth distribution:")
    for b in sorted(depth_counts.keys()):
        pct = (depth_counts[b] / sampled_rows) * 100.0
        print(f"  Depth [{b:2d} - {b+4:2d}]: {depth_counts[b]:>8,} ({pct:>5.1f}%)")

    return depth_counts


def process_parquet_file(
    input_path: str,
    output_writer: pq.ParquetWriter,
    min_depth: int = 15,
    max_positions: int = 60_000_000,
    current_count: int = 0,
) -> int:
    """Streams row groups from input parquet and writes filtered (fen, cp, depth) to writer."""
    pf = pq.ParquetFile(input_path)
    num_rgs = pf.num_row_groups
    count = current_count

    print(f"\nProcessing {input_path} (min_depth={min_depth}, max_positions={max_positions:,})...")
    t0 = time.time()

    for rg_idx in range(num_rgs):
        if count >= max_positions:
            print(f"Reached target cap of {max_positions:,} positions!")
            break

        table = pf.read_row_group(rg_idx, columns=["fen", "depth", "cp", "mate"])
        fens = table["fen"].to_pylist()
        depths = table["depth"].to_pylist()
        cps = table["cp"].to_pylist()
        mates = table["mate"].to_pylist()

        filtered_fens = []
        filtered_cps = []
        filtered_depths = []

        for fen, depth, cp, mate in zip(fens, depths, cps, mates):
            if depth is None or depth < min_depth:
                continue

            if cp is None and mate is None:
                continue

            # Determine side to move:
            parts = fen.split(" ")
            if len(parts) < 2:
                continue
            is_black_to_move = (parts[1] == "b")

            if cp is not None:
                val = max(min(cp, 30000), -30000)
                stm_cp = -val if is_black_to_move else val
            else:
                mate_val = 30000 if mate > 0 else -30000
                stm_cp = -mate_val if is_black_to_move else mate_val

            filtered_fens.append(fen)
            filtered_cps.append(stm_cp)
            filtered_depths.append(depth)

            count += 1
            if count >= max_positions:
                break

        if filtered_fens:
            batch_table = pa.Table.from_arrays(
                [
                    pa.array(filtered_fens, type=pa.string()),
                    pa.array(filtered_cps, type=pa.int16()),
                    pa.array(filtered_depths, type=pa.uint8()),
                ],
                names=["fen", "cp", "depth"],
            )
            output_writer.write_table(batch_table)

        elapsed = time.time() - t0
        rate = count / max(elapsed, 0.001)
        if (rg_idx + 1) % 5 == 0 or count >= max_positions or (rg_idx + 1) == num_rgs:
            print(
                f"  RG {rg_idx + 1:2d}/{num_rgs} | Extracted: {count:>10,} positions | "
                f"Rate: {rate:>9.0f} pos/sec | Elapsed: {elapsed:>5.1f}s"
            )

    return count


def main():
    parser = argparse.ArgumentParser(description="Filter and extract Lichess evaluations")
    parser.add_argument("--inputs", nargs="+", default=["data/data_0000.parquet"], help="Path to input parquet files")
    parser.add_argument("--output", default="data/lichess_filtered.parquet", help="Path to output parquet")
    parser.add_argument("--min_depth", type=int, default=16, help="Minimum search depth threshold")
    parser.add_argument("--min_positions", type=int, default=30_000_000, help="Minimum target positions")
    parser.add_argument("--max_positions", type=int, default=60_000_000, help="Maximum target positions")
    args = parser.parse_args()

    schema = pa.schema([
        ("fen", pa.string()),
        ("cp", pa.int16()),
        ("depth", pa.uint8()),
    ])

    total = 0
    with pq.ParquetWriter(args.output, schema, compression="zstd") as writer:
        for inp in args.inputs:
            if not os.path.exists(inp):
                print(f"Skipping missing file: {inp}")
                continue
            analyze_depth_distribution(inp, sample_row_groups=3)
            total = process_parquet_file(
                inp,
                writer,
                min_depth=args.min_depth,
                max_positions=args.max_positions,
                current_count=total,
            )
            if total >= args.max_positions:
                break

    out_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    print(f"\nExtraction complete!")
    print(f"Total positions written: {total:,}")
    print(f"Output file size: {out_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
