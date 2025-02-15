import subprocess
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Pytest command generator")
    parser.add_argument("--config-file", type=str, help="Load arguments from a configuration file.")
    args = parser.parse_args()

    config_file = args.config_file or "flag_gems.json"

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    working_dir = config["working_dir"]
    command_list_file = config["command_list_file"]
    excel_output_file = command_list_file.replace("commands.txt", "commands_excel.txt")

    cmd = ["pytest", "--collect-only", "--quiet", working_dir]
    process = subprocess.run(
        cmd, capture_output=True, text=True
    )

    collected_tests = set()

    with open(command_list_file, "w", encoding="utf-8") as f, open(excel_output_file, "w", encoding="utf-8") as excel_f:
        excel_f.write("Kernel Name\tTest Case Name\n")  # Write Excel header
        for line in process.stdout.splitlines():
            if "tests collected in" in line:
                continue
            if "=============================== warnings summary ===============================" in line:
                break
            if line.strip():
                # Remove parameterization details by splitting at the first '['
                test_identifier = line.split("[")[0]
                if "/" in test_identifier:
                    test_identifier = test_identifier.split("/", 1)[1]
                collected_tests.add(test_identifier)

        for test in sorted(collected_tests):
            command = f"pytest -s {test}"
            print(command)
            f.write(command + "\n")

            # Generate Excel-compatible output
            if "::" in test:
                kernel_name, test_case_name = test.split("::", 1)
                kernel_name = kernel_name.replace(".py", "")  # Remove .py suffix
                excel_f.write(f"{kernel_name}\t{test_case_name}\n")

if __name__ == "__main__":
    main()
