import glob
import json
import numpy as np
import shutil
import os
import subprocess

CONFIG_FILE = "config_train.yaml" # Your main config file
RESULTS_DIR = "random_partition_results" # Top-level dir for all runs
N_ITERATIONS = 10  # Set to a feasible number (e.g., 5 or 10)
KEY_METRIC = "mR@50" # The metric you want to compare from the JSON

def parse_metric_from_log_dir(log_dir, key_metric):
    try:
        # 1. Search recursively for all .json files in the run's log dir.
        glob_pattern = os.path.join(log_dir, "**", "*.json")
        all_json_files = glob.glob(glob_pattern, recursive=True)

        if not all_json_files:
            print(f"  Warning: No JSON files found in {log_dir}")
            return None

        # 2. Filter out the saved config file.
        metric_files = [
            f for f in all_json_files if not f.endswith("config_train.yaml")
        ]

        if not metric_files:
            print(f"  Warning: No metric JSON found (only config) in {log_dir}")
            return None

        metric_file_path = metric_files[0]
        print(f"  Found metric file: {metric_file_path}")

        with open(metric_file_path, 'r') as f:
            metrics = json.load(f)

        if isinstance(metrics, list):
            metrics_dict = metrics[0]
        else:
            metrics_dict = metrics

        score = metrics_dict.get(key_metric)

        if score is None:
            print(f"  Warning: Metric '{key_metric}' not in {metric_file_path}")
        return score

    except Exception as e:
        print(f"  Error parsing metrics from {log_dir}: {e}")
        return None

def run_training_experiment(seed, output_path, config_file):
    """
    Runs a single training experiment using subprocess.
    """
    # Clean up output path if it exists, to ensure a fresh run
    if os.path.exists(output_path):
        print(f"  Removing old directory: {output_path}")
        shutil.rmtree(output_path)

    # Build the command
    cmd = [
        "python", "train_egtr.py",
        "--config", config_file,
        "--output_path", output_path,
        "--resume=False",
        "--skip_train=False",
        "--eval_when_train_end=True"
    ]

    # Add the seed argument only if it's for a random partition
    if seed is not None:
        cmd.extend(["--random_partition_seed", str(seed)])
        print(f"Running experiment with random_partition_seed={seed}...")
    else:
        print("Running experiment with MANUAL partition...")

    try:
        # Run the command
        subprocess.run(cmd, check=True)

        # After training, parse results from the output path
        print(f"  Run complete. Parsing results from {output_path}...")
        return parse_metric_from_log_dir(output_path, KEY_METRIC)

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Training run failed for seed {seed}.")
        print(f"  Command was: {' '.join(cmd)}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        return None

def main():
    """
    Main experiment runner.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Run the manual partition (seed=None)
    manual_output_path = os.path.join(RESULTS_DIR, "run_manual")
    manual_score = run_training_experiment(
        seed=None,
        output_path=manual_output_path,
        config_file=CONFIG_FILE
    )

    if manual_score is None:
        print("Manual run failed or metric not found. Aborting.")
        return

    print(f"\n--- MANUAL PARTITION SCORE: {manual_score:.4f} ---\n")

    random_scores = []
    for i in range(N_ITERATIONS):
        print(f"--- STARTING RANDOM RUN {i+1}/{N_ITERATIONS} ---")
        random_output_path = os.path.join(RESULTS_DIR, f"run_seed_{i}")
        score = run_training_experiment(
            seed=i,
            output_path=random_output_path,
            config_file=CONFIG_FILE
        )

        if score is not None:
            random_scores.append(score)
        print(f"--- COMPLETED RANDOM RUN {i+1}/{N_ITERATIONS} ---")

    print("\n--- EXPERIMENT COMPLETE ---")
    print(f"Manual Partition Score ({KEY_METRIC}): {manual_score:.4f}")

    if random_scores:
        mean_random = np.mean(random_scores)
        std_random = np.std(random_scores)

        print(f"Random Partition Scores ({len(random_scores)} runs): {random_scores}")
        print(f"Mean (μ_random): {mean_random:.4f}")
        print(f"Std Dev (σ_random): {std_random:.4f}")

    else:
        print("No successful random runs to compare against.")

if __name__ == "__main__":
    main()
