import glob
import json
import shutil
import os
import subprocess
import argparse
import sys

DEFAULT_CONFIG_FILE = "config_train.yaml"
DEFAULT_RESULTS_DIR = "random_partition_results"
KEY_METRIC = "(single)mR@50"


def parse_metric_from_log_dir(log_dir, key_metric):
    """
    Finds the final metric .json file in the log dir and parses it.
    """
    try:
        glob_pattern = os.path.join(log_dir, "**", "checkpoints", "*.json")
        all_metric_files = glob.glob(glob_pattern, recursive=True)

        if not all_metric_files:
            print(
                f"  Warning: No metric JSON (matching '{glob_pattern}') found in {log_dir}"
            )
            return None

        metric_file_path = all_metric_files[0]
        print(f"  Found metric file: {metric_file_path}")

        with open(metric_file_path, "r") as f:
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
        "python",
        "train_egtr.py",
        "--config",
        config_file,
        "--output_path",
        output_path,
        "--resume=False",
        "--skip_train=False",
        "--eval_when_train_end=True",
        "--random_partition_seed", str(seed)
    ]

    print(f"Running single experiment with random_partition_seed={seed}...")
    print(f"Output path: {output_path}")

    try:
        # Run the command
        subprocess.run(cmd, check=True, env=os.environ)

        # After training, parse results from the output path
        print(f"  Run complete. Parsing results from {output_path}...")
        return parse_metric_from_log_dir(output_path, KEY_METRIC)

    except subprocess.CalledProcessError:
        print(f"  ERROR: Training run failed for seed {seed}.")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run a single random partition experiment.")
    parser.add_argument("--seed", type=int, required=True, help="The random seed to use for this partition.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to config file.")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Base directory for results.")

    args = parser.parse_args()

    # Ensure base results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    # Define specific output path for this seed
    output_path = os.path.join(args.results_dir, f"run_seed_{args.seed}")

    # Run the single experiment
    score = run_training_experiment(
        seed=args.seed, 
        output_path=output_path, 
        config_file=args.config
    )

    if score is not None:
        print(f"\n--- SUCCESS: Seed {args.seed} finished ---")
        print(f"{KEY_METRIC}: {score:.4f}")
    else:
        print(f"\n--- FAILED: Seed {args.seed} did not return a valid score ---")
        sys.exit(1)

if __name__ == "__main__":
    main()
