#!/usr/bin/env python3
"""
utils.py - Scaffolding code for running the submission.
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from params import InstanceParams, SINGLE, LARGE
from typing import Tuple

# Global variable to track the last timestamp
_last_timestamp: datetime = None
# Global variable to store measured times
_timestamps = {}
_timestampsStr = {}
# Global variable to store measured sizes
_bandwidth = {}
# Global variable to store model quality metrics
_model_quality = {}

def parse_submission_arguments(workload: str) -> Tuple[int, InstanceParams, int, int, int, bool, str, str]:
    """
    Get the arguments of the submission. Populate arguments as needed for the workload.
    """
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description=workload)
    parser.add_argument('size', type=int, choices=range(SINGLE, LARGE+1),
                        help='Instance size (0-single/1-small/2-medium/3-large)')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of times to run steps 4-9 (default: 1)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for dataset and query generation')
    parser.add_argument('--clrtxt', type=int,
                        help='Specify with 1 if to rerun the cleartext computation')
    parser.add_argument('--remote', action='store_true',
                        help='Run example submission in remote backend mode')
    parser.add_argument('--model', default=None, type=str,
                        help='Pick a model run (default: mlp)')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='Pick a dataset run (default: mnist)')
    

    args = parser.parse_args()
    size = args.size
    seed = args.seed
    num_runs = args.num_runs
    clrtxt = args.clrtxt
    remote_be = args.remote

    # adding model and dataset to the arguments
    dataset_name = args.dataset.lower()

    # Dataset-specific model defaults
    dataset_model_defaults = {
        "mnist": "mlp",
        "cifar10": "resnet20",
    }
    
    # Model selection: use provided model or dataset-specific default
    if args.model is None:
        model_name = dataset_model_defaults.get(dataset_name, "mlp")
    else:
        model_name = args.model.lower()

    # Use params.py to get instance parameters
    params = InstanceParams(size)
    return size, params, seed, num_runs, clrtxt, remote_be, model_name, dataset_name

def ensure_directories(rootdir: Path):
    """ Check that the current directory has sub-directories
    'harness', 'scripts', and 'submissions' """
    required_dirs = ['harness', 'scripts', 'submissions']
    for dir_name in required_dirs:
        if not (rootdir / dir_name).exists():
            print(f"Error: Required directory '{dir_name}'",
                  f"not found in {rootdir}")
            sys.exit(1)

def build_submission(script_dir: Path, model_name: str, remote_be: bool):
    """
    Build the submission, including pulling dependencies as neeed
    """
    if remote_be:
        subprocess.run(["pip", "install", "-r", f"./submission_remote/{model_name}/requirements.txt"], check=True)
    else:
        # Clone and build OpenFHE if needed
        subprocess.run([script_dir/"get_openfhe.sh"], check=True)
        # CMake build of the submission itself
        subprocess.run([script_dir/"build_task.sh", f"./submissions/{model_name}"], check=True)

class TextFormat:
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    RED = "\033[31m"
    RESET = "\033[0m"

def log_step(step_num: int, step_name: str, start: bool = False):
    """ 
    Helper function to print timestamp after each step with second precision 
    """
    global _last_timestamp
    global _timestamps
    global _timestampsStr
    now = datetime.now()
    # Format with milliseconds precision
    timestamp = now.strftime("%H:%M:%S")

    # Calculate elapsed time if this isn't the first call
    elapsed_str = ""
    elapsed_seconds = 0
    if _last_timestamp is not None:
        elapsed_seconds = (now - _last_timestamp).total_seconds()
        elapsed_str = f" (elapsed: {round(elapsed_seconds, 4)}s)"

    # Update the last timestamp for the next call
    _last_timestamp = now

    if (not start):
        print(f"{TextFormat.BLUE}{timestamp} [harness] {step_num}: {step_name} completed{elapsed_str}{TextFormat.RESET}")
        _timestampsStr[step_name] = f"{round(elapsed_seconds, 4)}s"
        _timestamps[step_name] = elapsed_seconds

def log_size(path: Path, object_name: str, flag: bool = False, previous: int = 0):
    global _bandwidth
    
    # Check if the path exists before trying to calculate size
    if not path.exists():
        print(f"         [harness] Warning: {object_name} path does not exist: {path}")
        _bandwidth[object_name] = "0B"
        return 0
    
    size = int(subprocess.run(["du", "-sb", path], check=True,
                           capture_output=True, text=True).stdout.split()[0])
    if(flag):
        size -= previous
    
    print(f"{TextFormat.YELLOW}         [harness] {object_name} size: {human_readable_size(size)}{TextFormat.RESET}")

    _bandwidth[object_name] = human_readable_size(size)
    return size

def human_readable_size(n: int):
    for unit in ["B","K","M","G","T"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"

def save_run(path: Path, submission_report_path: Path, model_name: str, dataset_name: str, size: int = 0):
    global _timestamps
    global _timestampsStr
    global _bandwidth
    global _model_quality

    _timestampsStr["Total"] = f"{round(sum(_timestamps.values()), 4)}s"
    _timestampsReported = {}
    if submission_report_path.exists():
        with open(submission_report_path, "r") as f:
            server_reported_times = json.load(f)
            print(f"{TextFormat.GREEN}         [submission] Server reported steps: {server_reported_times}{TextFormat.RESET}")
            for step_name, time_str in server_reported_times.items():
                _timestampsReported[step_name] = f"{time_str}s"
                print(f"{TextFormat.PURPLE}         [submission] {step_name}: {time_str}s{TextFormat.RESET}")
    else:
        print(f"{TextFormat.PURPLE}         [harness] Note: Submitters can specify Server reported steps file at {submission_report_path}{TextFormat.RESET}")

    if size == 0:
        json.dump({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "Timing": _timestampsStr,
            "Bandwidth": _bandwidth,
            "Server Reported": _timestampsReported,
        }, open(path, "w"), indent=2)
    else:
        json.dump({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "Timing": _timestampsStr,
            "Bandwidth": _bandwidth,
            "Quality": _model_quality,
            "Server Reported": _timestampsReported,
        }, open(path, "w"), indent=2)

    print("[total latency]", f"{round(sum(_timestamps.values()), 4)}s")

def calculate_quality(label_file: Path, pred_file: Path, tag: str):
    """
    Calculates accuracy by comparing labels line by line.
    Label file and predictions file should contain one label per line.
    Logs accuracy metric and prints results.
    """

    try:
        # Read expected labels (one per line)
        labels = label_file.read_text().strip().split('\n')
        labels = [label.strip() for label in labels if label.strip()]

        # Read result labels (one per line)
        preds = pred_file.read_text().strip().split('\n')
        preds = [label.strip() for label in preds if label.strip()]

    except Exception as e:
        print(f"[harness] failed to read files: {e}")
        sys.exit(1)

    num_samples = len(preds)

    correct_pred = sum(1 for exp, res in zip(labels, preds) if exp == res)
    accuracy = correct_pred / num_samples
    print(f"[harness] {tag}: {accuracy:.4f} ({correct_pred}/{num_samples} correct)")
    log_quality(correct_pred, num_samples, f"{tag} quality")


def log_quality(correct_predictions, total_samples, tag):
    global _model_quality
    _model_quality[tag] = {
        "correct_predictions": correct_predictions,
        "total_samples": total_samples,
        "accuracy": correct_predictions / total_samples if total_samples > 0 else 0
    }

def run_exe_or_python(base, file_name, *args, check=True):
    """
        If {base}/{file_name}.py exists, run it with the current Python.
        Otherwise, run {base}/build/{file_name} as an executable.
    """
    py = base / f"{file_name}.py"
    exe = base / "build" / file_name

    if py.exists():
        cmd = ["python3", py, *args]
    elif exe.exists():
        cmd = [exe, *args]
    else:
        cmd = None
    if cmd is not None:
        subprocess.run(cmd, check=check)
