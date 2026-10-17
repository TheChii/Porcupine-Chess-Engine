"""
FH-NNUE Comprehensive Experimental Suite

Runs the architecture ablations:
- B0: conventional unfactorized HalfKAv2
- A1: horizontal canonicalization only
- A2: global + local factorization
- A3: global + relative-displacement + zone + local factorization
- A4: PSQT skip connection on/off
- Data-efficiency scaling comparisons (10%, 25%, 100%)
"""

import os
import json
from train import run_experiment


def main():
    experiments = [
        {"name": "B0_baseline", "arch": "B0", "has_psqt": False, "fraction": 1.0},
        {"name": "A1_canonical", "arch": "A1", "has_psqt": False, "fraction": 1.0},
        {"name": "A2_factorized_global_local", "arch": "A2", "has_psqt": False, "fraction": 1.0},
        {"name": "A3_factorized_full", "arch": "A3", "has_psqt": False, "fraction": 1.0},
        {"name": "A4_full_plus_psqt", "arch": "A3", "has_psqt": True, "fraction": 1.0},
        # Data efficiency scaling with A3 + PSQT vs B0 baseline at 10% and 25%
        {"name": "B0_data_10pct", "arch": "B0", "has_psqt": False, "fraction": 0.10},
        {"name": "A3_data_10pct", "arch": "A3", "has_psqt": True, "fraction": 0.10},
        {"name": "B0_data_25pct", "arch": "B0", "has_psqt": False, "fraction": 0.25},
        {"name": "A3_data_25pct", "arch": "A3", "has_psqt": True, "fraction": 0.25},
    ]

    results = []
    print("Starting FH-NNUE Experimental Suite...\n")

    for exp in experiments:
        out_name = f"model_{exp['name']}.nnue"
        res = run_experiment(
            arch_type=exp["arch"],
            has_psqt=exp["has_psqt"],
            data_fraction=exp["fraction"],
            epochs=2,
            batch_size=64,
            output_model_path=out_name,
        )
        res["name"] = exp["name"]
        results.append(res)
        print("-" * 60)

    # Save results as JSON
    with open("experiments_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table
    print("\n\n" + "=" * 80)
    print("                    EXPERIMENTAL RESULTS SUMMARY                    ")
    print("=" * 80)
    print(f"| {'Experiment':<28} | {'Arch':<5} | {'PSQT':<5} | {'Data %':<6} | {'Val Loss':<8} | {'Test Loss':<9} | {'Test MSE':<8} | {'Time (s)':<8} |")
    print(f"|{'-'*30}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*10}|{'-'*11}|{'-'*10}|{'-'*10}|")
    for r in results:
        print(
            f"| {r['name']:<28} | {r['arch_type']:<5} | {str(r['has_psqt']):<5} | {r['data_fraction']*100:>5.0f}% | {r['val_loss']:>8.4f} | {r['test_loss']:>9.4f} | {r['test_mse']:>8.4f} | {r['training_time_s']:>8.2f} |"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
