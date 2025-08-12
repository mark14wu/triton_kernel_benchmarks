import os
import subprocess
from pathlib import Path
import time
import json
import hashlib

def compute_md5(file_path):
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def parse_command_line(cmd_line):
    """
    Parse a command line (e.g., 'pytest -s test_ops.py::test_something') to extract kernel_name and test_case_name.
    If it doesn't match the pattern, return the entire line as kernel_name and empty test_case_name.
    """
    # Default
    kernel_name = cmd_line
    test_case_name = ""

    # Simple parse for 'pytest' commands
    if "pytest" in cmd_line:
        # Try splitting by '::'
        parts = cmd_line.split("::", 1)
        if len(parts) == 1:
            print(parts)
            raise ValueError(f"Invalid pytest command line: {cmd_line}, expected 'pytest -s <kernel_name>.py::<test_case_name>'")
        left, right = parts
        test_case_name = right.strip()

        # Now remove '.py' from the left part if present
        left = left.strip()
        if ".py" in left:
            left = left.split(".py")[0]
        kernel_name = left.replace("pytest -s", "").replace("pytest", "").strip()
    return kernel_name, test_case_name

def load_tsv_results(tsv_file):
    """
    Load existing results from a TSV file into a dictionary:
    results_dict[(kernel_name, test_case_name)] = {
        'baseline': str_time or 'n/a' or 'skipped:...' or 'failed',
        'compute-sanitizer': ...,
        'z3-sanitizer': ...
    }
    We only track these three prefixes as per requirement.
    """
    results = {}
    if not os.path.exists(tsv_file):
        return results

    with open(tsv_file, "r") as f:
        # Skip header
        header = f.readline().strip().split("\t")
        # Expecting: kernel_name, test_case_name, baseline_time, compute_sanitizer_time, z3_time
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            kn, tcn, baseline_t, compute_sanitizer_t, z3_t = cols
            results[(kn, tcn)] = {
                "baseline": baseline_t,
                "compute-sanitizer": compute_sanitizer_t,
                "z3-sanitizer": z3_t
            }
    return results

def save_tsv_results(tsv_file, results_dict):
    """
    Save the results to a TSV file.
    Columns: kernel_name, test_case_name, baseline_time, compute_sanitizer_time, z3_time
    """
    with open(tsv_file, "w") as f:
        f.write("kernel_name\ttest_case_name\tbaseline_time\tcompute_sanitizer_time\tz3_time\n")
        for (kn, tcn), time_map in results_dict.items():
            baseline_time = time_map.get("baseline", "n/a")
            compute_sanitizer_time = time_map.get("compute-sanitizer", "n/a")
            z3_time = time_map.get("z3-sanitizer", "n/a")
            f.write(f"{kn}\t{tcn}\t{baseline_time}\t{compute_sanitizer_time}\t{z3_time}\n")

def run_commands(command_list_file, output_dir, working_dir, selected_prefixes):
    """Run commands with different prefixes, handle logging, and produce/update a TSV file with results."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Progress log file
    progress_log_file = Path(output_dir) / "progress.log"
    completed_commands = set()

    # Load completed commands from progress log
    if progress_log_file.exists():
        with open(progress_log_file, "r") as log:
            completed_commands = set(line.strip() for line in log if line.strip())

    # Read command list from file
    with open(command_list_file, "r") as f:
        raw_lines = f.readlines()

    commands = []
    for line in raw_lines:
        line = line.rstrip("\n")
        if not line.strip():
            # Skip empty lines
            continue
        commands.append(line)

    # Define the different prefixes and their environment setup
    prefixes = {
        "baseline": "TRITON_SANITIZER_BACKEND=off TRITON_ALWAYS_COMPILE=1 ",
        "baseline-amd": "TRITON_SANITIZER_BACKEND=off TRITON_ALWAYS_COMPILE=1 ",
        "compute-sanitizer": "TRITON_ALWAYS_COMPILE=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer ",
        "llvm-sanitizer": "TRITON_ENABLE_ASAN=1 TRITON_ALWAYS_COMPILE=1 ",
        "z3-sanitizer": "TRITON_SANITIZER_BACKEND=symexec ",

    }
    prefix_env_setup = {
        "baseline-amd": "source /root/triton/bin/activate",
        "compute-sanitizer": "deactivate && source /home/hwu27/workspace/venv/triton_cuda_12_2/bin/activate && source /etc/profile.d/modules.sh && module load cuda/12.2",
        "llvm-sanitizer": "source /root/triton/bin/activate"
    }

    # Process comma-separated prefixes into a list
    if isinstance(selected_prefixes, str):
        selected_prefixes = [p.strip() for p in selected_prefixes.split(',')]

    # Filter prefixes based on selection
    if "all" in selected_prefixes:
        selected_prefixes = list(prefixes.keys())
    else:
        for sp in selected_prefixes:
            if sp not in prefixes:
                raise ValueError(f"Invalid prefix: {sp}. Choose from {list(prefixes.keys())} or 'all'.")

    # Prepare the TSV file paths
    tsv_file_path = os.path.join(output_dir, "results.tsv")
    md5_file_path = os.path.join(output_dir, "results.md5")

    # 1) If TSV exists, verify MD5
    if os.path.exists(tsv_file_path) and os.path.exists(md5_file_path):
        old_md5 = None
        with open(md5_file_path, "r") as mdf:
            old_md5 = mdf.read().strip()
        current_md5 = compute_md5(tsv_file_path)
        if current_md5 != old_md5:
            raise RuntimeError("TSV file has been modified externally (MD5 mismatch). Aborting for safety.")

    # 2) Load existing results
    results_dict = load_tsv_results(tsv_file_path)

    # Total commands for progress reporting
    normal_commands_count = sum(1 for c in commands if not c.strip().startswith("#"))
    total_commands = normal_commands_count * len(selected_prefixes)
    command_counter = 0

    # Execute commands in the specified order
    for line in commands:
        # Check if it's a commented line
        if line.startswith("#"):
            # We still create a row in the TSV, marking times as "skipped"
            # Possibly parse reason after the '#' and the actual command after a ':'
            line_content = line[1:].strip()
            if ":" in line_content:
                reason_part, real_cmd = line_content.split(":", 1)
                reason_part = reason_part.strip()
                real_cmd = real_cmd.strip()
                # Parse kernel_name/test_case_name from the real command
                kn, tcn = parse_command_line(real_cmd)
                skip_reason = reason_part
            else:
                # No explicit reason, just skip
                kn, tcn = parse_command_line(line_content)
                skip_reason = ""

            if (kn, tcn) not in results_dict:
                results_dict[(kn, tcn)] = {
                    "baseline": f"{skip_reason}" if skip_reason else "skipped",
                    "compute-sanitizer": f"{skip_reason}" if skip_reason else "skipped",
                    "z3-sanitizer": f"{skip_reason}" if skip_reason else "skipped"
                }
            else:
                # If it already exists, we forcibly set them to "skipped", in case it wasn't set before
                for p_col in ["baseline", "compute-sanitizer", "z3-sanitizer"]:
                    results_dict[(kn, tcn)][p_col] = f"{skip_reason}" if skip_reason else "skipped"
            # We don't run anything for a commented line, just continue
            continue

        # If it's a normal command:
        kn, tcn = parse_command_line(line)

        # Make sure there's an entry in results_dict
        if (kn, tcn) not in results_dict:
            # Initialize with 'n/a' if there's no prior record
            results_dict[(kn, tcn)] = {}
            for prefix_key in selected_prefixes:
                results_dict[(kn, tcn)][prefix_key] = "n/a"

        # For each prefix
        for prefix_key in selected_prefixes:
            command_counter += 1
            formatted_command_counter = str(command_counter).zfill(len(str(total_commands)))
            progress = f"[{command_counter}/{total_commands}]" if normal_commands_count > 0 else ""

            full_cmd = f"{prefixes[prefix_key]}{line}"

            # Check if command is already completed
            if full_cmd in completed_commands:
                print(f"Skipping {progress}: {full_cmd} (already completed)")
                continue

            print(f"Running {progress}: Prefix: '{prefix_key}', Command: '{line}'")

            output_file = Path(output_dir) / f"{formatted_command_counter}_{full_cmd.replace(' ', '_').replace('::', '_').replace('/', '_')}.log"
            start_time = time.time()

            # Run command
            if prefix_key in prefix_env_setup:
                env_command = prefix_env_setup[prefix_key]
                process = subprocess.Popen(
                    f"bash -c 'source ~/.bashrc && {env_command} && {full_cmd}'",
                    shell=True, cwd=working_dir, stdout=open(output_file, "w"), stderr=subprocess.STDOUT
                )
            else:
                process = subprocess.Popen(
                    f"bash -c 'source ~/.bashrc && {full_cmd}'",
                    shell=True, cwd=working_dir, stdout=open(output_file, "w"), stderr=subprocess.STDOUT
                )
            process.wait()
            elapsed_time = time.time() - start_time

            # Check if command executed successfully
            if process.returncode == 0:
                print(f"Completed {progress}: Prefix: '{prefix_key}', Command: '{line}' in {elapsed_time:.2f}s")
                # Update TSV with time
                results_dict[(kn, tcn)][prefix_key] = f"{elapsed_time:.4f}"
                # Save progress to log
                with open(progress_log_file, "a") as log:
                    log.write(full_cmd + "\n")
            else:
                print(f"Failed {progress}: Prefix: '{prefix_key}', Command: '{line}'")
                # Update TSV with "failed"
                results_dict[(kn, tcn)][prefix_key] = "failed"

            # After each command, save the TSV
            save_tsv_results(tsv_file_path, results_dict)

            # Compute new MD5 and save
            new_md5 = compute_md5(tsv_file_path)
            with open(md5_file_path, "w") as mdf:
                mdf.write(new_md5)

if __name__ == "__main__":
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-list-file", type=str, default="commands.txt", help="File containing list of commands.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save output logs.")
    parser.add_argument("--working-dir", type=str, default="../FlagAttention/tests/flag_attn", help="Working directory to run commands.")
    parser.add_argument("--selected-prefixes", type=str, default="all", help="Choose prefixes to run commands with (default: all). Use comma-separated values like 'baseline,compute-sanitizer'.")
    parser.add_argument("--config-file", type=str, help="Load arguments from a configuration file.")

    args = parser.parse_args()

    # Load arguments from config file if provided
    if args.config_file:
        if os.path.exists(args.config_file):
            with open(args.config_file, "r") as f:
                config_args = json.load(f)
            for key, value in config_args.items():
                # Only set attributes that are not already set via command line
                if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)
        else:
            print(f"Warning: Config file {args.config_file} not found. Skipping.")

    # Expand the working directory path
    working_dir = os.path.expanduser(args.working_dir)

    # Run the script
    run_commands(args.command_list_file, args.output_dir, working_dir, args.selected_prefixes)
