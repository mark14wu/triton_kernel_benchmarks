import argparse
from os import error
import triton_viz
from triton_viz.clients.sanitizer.data import OutOfBoundsRecord
from triton_viz.core.trace import launches
import numpy as np

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

def check_out_of_bounds():
    assert len(launches) > 0, "No launches found, make sure you run triton kernel with @triton_viz.trace!"
    error_msgs = []
    for launch in launches:
        for record in launch.records:
            if isinstance(record, OutOfBoundsRecord):
                result_invalid_masks = record.invalid_access_masks
                if np.any(result_invalid_masks):
                    non_false_indices = np.where(result_invalid_masks)
                    error_msg = f"Found out-of-bound error at indices: {non_false_indices}\n"
                    error_msg += f"Total indices: {len(result_invalid_masks)}\n"
                    error_msg += 60 * "=" + "\n"
                    error_msgs.append(error_msg)                              
    
    if error_msgs:
        print(20 * '=' + 'Out-of-bound error detected!' + 20 * '=')
        for error_msg in error_msgs:
            print(error_msg)
        assert False

    triton_viz.clear()