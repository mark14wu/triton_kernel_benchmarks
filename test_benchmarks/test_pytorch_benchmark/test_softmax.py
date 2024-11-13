import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name


@pytest.mark.parametrize("iter", range(98))
def test_triton_softmax(iter):
    Operator = load_opbench_by_name('softmax')
    opbench = Operator(tb_args=parse_torchbench_args())

    x = opbench.get_example_inputs()[0]

    ans = opbench.triton_softmax(x)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'
