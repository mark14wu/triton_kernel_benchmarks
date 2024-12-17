import os
import subprocess
from pathlib import Path
import time
import json

def run_commands(command_list_file, output_dir, working_dir, selected_prefixes, dryrun=False):
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
        commands = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    # Define the different prefixes and their environment setup
    prefixes = {
        "baseline": "",
        "compute-sanitizer": "PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer ",
        "triton-sanitizer": "TRITON_SANITIZER_BACKEND=brute_force "
    }
    prefix_env_setup = {
        "compute-sanitizer": "source /etc/profile.d/modules.sh && module load cuda/12.2"
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

    # Total commands for progress reporting
    total_commands = len(commands) * len(selected_prefixes)
    command_counter = 0

    # Execute commands in the specified order
    for prefix_key in selected_prefixes:
        prefix = prefixes[prefix_key]
        for cmd in commands:
            full_cmd = f"{prefix}{cmd}"
            output_file = Path(output_dir) / f"{full_cmd.replace(' ', '_').replace('::', '_').replace('/', '_')}.log"

            # Check if command is already completed
            if full_cmd in completed_commands:
                print(f"Skipping: {full_cmd} (already completed)")
                continue

            command_counter += 1
            progress = f"[{command_counter}/{total_commands}]"

            if dryrun:
                print(f"Dryrun {progress}: Prefix: '{prefix_key}', Command: '{cmd}' -> Output: {output_file}")
            else:
                print(f"Running {progress}: Prefix: '{prefix_key}', Command: '{cmd}'")

                # Run the command and save output
                with open(output_file, "w") as outfile:
                    start_time = time.time()
                    # Setup environment for specific prefixes
                    if prefix_key in prefix_env_setup:
                        env_command = prefix_env_setup[prefix_key]
                        process = subprocess.Popen(f"bash -c 'source ~/.bashrc && {env_command} && {full_cmd}'", shell=True, cwd=working_dir, stdout=outfile, stderr=subprocess.STDOUT)
                    else:
                        process = subprocess.Popen(f"bash -c 'source ~/.bashrc && {full_cmd}'", shell=True, cwd=working_dir, stdout=outfile, stderr=subprocess.STDOUT)
                    process.wait()
                    elapsed_time = time.time() - start_time

                # Check if command executed successfully
                if process.returncode == 0:
                    with open(progress_log_file, "a") as log:
                        log.write(full_cmd + "\n")
                    print(f"Completed {progress}: Prefix: '{prefix_key}', Command: '{cmd}' in {elapsed_time:.2f}s")
                else:
                    print(f"Failed {progress}: Prefix: '{prefix_key}', Command: '{cmd}'")
                    return

if __name__ == "__main__":
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run a list of commands with optional dryrun support and prefix control.")
    parser.add_argument("--command-list-file", type=str, default="commands.txt", help="File containing list of commands.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save output logs.")
    parser.add_argument("--working-dir", type=str, default="/home/hwu27/workspace/triton_kernel_benchmarks/FlagAttention/tests/flag_attn", help="Working directory to run commands.")
    parser.add_argument("--selected-prefixes", type=str, default="all", help="Choose prefixes to run commands with (default: all). Use comma-separated values like 'baseline,compute-sanitizer'.")
    parser.add_argument("--dryrun", action="store_true", help="If set, only print the commands without executing them.")
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
    run_commands(args.command_list_file, args.output_dir, working_dir, args.selected_prefixes, dryrun=args.dryrun)
