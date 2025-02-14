import torchbenchmark.models.nvidia_deeprecommender
# import torchbenchmark.canary_models.torchrec_dlrm

def test_nvidia_deeprecommender():
    model, example_inputs = torchbenchmark.models.nvidia_deeprecommender.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

# def test_torchrec_dlrm():
#     model, example_inputs = torchbenchmark.models.torchrec_dlrm.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)
