import argparse


def parse_torchbench_args():
    tb_args = argparse.Namespace()

    # Test mode (fwd, bwd, or fwd_bwd).
    tb_args.mode = "fwd"

    # Device to benchmark.
    tb_args.device = "cuda"

    # Metrics to collect, split with comma. E.g., --metrics latency,tflops,speedup.
    tb_args.metrics = None

    # Override default baseline.
    tb_args.baseline = None

    # Specify one or multiple operator implementations to run.
    tb_args.only = None

    # Specify the start input id to run.
    # For example, --input-id 0 runs only the first available input sample.
    # When used together like --input-id <X> --num-inputs <Y>, start from the input id <X>
    # and run <Y> different inputs.
    tb_args.input_id = 0

    # Number of example inputs.
    tb_args.num_inputs = 1

    # idk what is this field used for...
    tb_args.keep_going = True

    tb_args.test_only = True

    return tb_args