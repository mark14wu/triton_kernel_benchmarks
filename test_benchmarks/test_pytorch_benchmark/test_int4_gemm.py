# triton.runtime.errors.OutOfResources: out of resource: 
# shared memory, Required: 139264, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.


# import pytest
# from benchmark_utils import parse_torchbench_args
# from torchbenchmark.operators import load_opbench_by_name
# import torch

# @pytest.mark.parametrize("iter", range(32))
# def test_triton(iter):
#     Operator = load_opbench_by_name('int4_gemm')
#     opbench = Operator(tb_args=parse_torchbench_args())
    
#     x, w = opbench.get_example_inputs()

#     ans = opbench.triton(x, w)()

#     assert len(ans) == 1
#     ans = ans[0]
#     assert ans.device.type == 'cuda'


