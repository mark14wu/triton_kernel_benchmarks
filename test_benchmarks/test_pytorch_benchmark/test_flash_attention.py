import pytest
from benchmark_utils import parse_torchbench_args
from triton_viz.clients import Sanitizer
from torchbenchmark.operators import load_opbench_by_name


@pytest.mark.parametrize("iter", range(6))
def test_flash_attention_triton_op_flash_seq_v2(iter):
    Operator = load_opbench_by_name('flash_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    q, k, v = opbench.get_example_inputs()
    assert q is not None, "q is None"
    assert k is not None, "k is None"
    assert v is not None, "v is None"

    ans = opbench.triton_op_flash_seq_v2(q, k, v)()

    assert ans is not None, "ans is None"

@pytest.mark.parametrize("iter", range(6))
def test_flash_attention_triton_op_flash_v2(iter):
    Operator = load_opbench_by_name('flash_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    q, k, v = opbench.get_example_inputs()
    assert q is not None, "q is None"
    assert k is not None, "k is None"
    assert v is not None, "v is None"

    ans = opbench.triton_op_flash_v2(q, k, v)()

    assert ans is not None, "ans is None"

@pytest.mark.parametrize("iter", range(6))
def test_flash_attention_triton_tutorial_flash_v2(iter):
    Operator = load_opbench_by_name('flash_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    q, k, v = opbench.get_example_inputs()
    assert q is not None, "q is None"
    assert k is not None, "k is None"
    assert v is not None, "v is None"

    ans = opbench.triton_tutorial_flash_v2(q, k, v)()

    assert ans is not None, "ans is None"

@pytest.mark.parametrize("iter", range(6))
def test_flash_attention_triton_tutorial_flash_v2_tma(iter):
    Operator = load_opbench_by_name('flash_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    q, k, v = opbench.get_example_inputs()
    assert q is not None, "q is None"
    assert k is not None, "k is None"
    assert v is not None, "v is None"

    ans = opbench.triton_tutorial_flash_v2_tma(q, k, v)()

    assert ans is not None, "ans is None"
