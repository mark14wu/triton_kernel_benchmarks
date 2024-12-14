import os
import subprocess
from pathlib import Path
import time

def run_commands(command_list_file, output_dir, working_dir, dryrun=False):
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
    prefixes = [
        "",
        "compute-sanitizer ",
        "TRITON_SANITIZER_BACKEND=brute_force "
    ]
    prefix_env_setup = {
        "compute-sanitizer": "source /etc/profile.d/modules.sh && module load cuda/12.2"
    }

    # Total commands for progress reporting
    total_commands = len(commands) * len(prefixes)
    command_counter = 0

    # Execute commands in the specified order
    for prefix in prefixes:
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
                print(f"Dryrun {progress}: Prefix: '{prefix.strip() or 'default'}', Command: '{cmd}' -> Output: {output_file}")
            else:
                print(f"Running {progress}: Prefix: '{prefix.strip() or 'default'}', Command: '{cmd}'")

                # Run the command and save output
                with open(output_file, "w") as outfile:
                    start_time = time.time()
                    # Setup environment for specific prefixes
                    if prefix.strip() in prefix_env_setup:
                        env_command = prefix_env_setup[prefix.strip()]
                        process = subprocess.Popen(f"bash -c 'source ~/.bashrc && {env_command} && {full_cmd}'", shell=True, cwd=working_dir, stdout=outfile, stderr=subprocess.STDOUT)
                    else:
                        process = subprocess.Popen(f"bash -c 'source ~/.bashrc && {full_cmd}'", shell=True, cwd=working_dir, stdout=outfile, stderr=subprocess.STDOUT)
                    process.wait()
                    elapsed_time = time.time() - start_time

                # Check if command executed successfully
                if process.returncode == 0:
                    with open(progress_log_file, "a") as log:
                        log.write(full_cmd + "\n")
                    print(f"Completed {progress}: Prefix: '{prefix.strip() or 'default'}', Command: '{cmd}' in {elapsed_time:.2f}s")
                else:
                    print(f"Failed {progress}: Prefix: '{prefix.strip() or 'default'}', Command: '{cmd}'")
                    return

if __name__ == "__main__":
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run a list of commands with optional dryrun support.")
    parser.add_argument("--command-list-file", type=str, default="commands.txt", help="File containing list of commands.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save output logs.")
    parser.add_argument("--working-dir", type=str, default="/home/hwu27/workspace/triton_kernel_benchmarks/FlagAttention/tests/flag_attn", help="Working directory to run commands.")
    parser.add_argument("--dryrun", action="store_true", help="If set, only print the commands without executing them.")

    args = parser.parse_args()

    # Expand the working directory path
    working_dir = os.path.expanduser(args.working_dir)

    # Run the script
    run_commands(args.command_list_file, args.output_dir, working_dir, dryrun=args.dryrun)
