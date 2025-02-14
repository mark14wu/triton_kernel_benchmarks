import torchbenchmark.models.timm_vision_transformer_large
import torchbenchmark.models.pytorch_CycleGAN_and_pix2pix
import torchbenchmark.models.dcgan
import torchbenchmark.models.timm_vision_transformer
import torchbenchmark.canary_models.DALLE2_pytorch
import torchbenchmark.models.pytorch_stargan

def test_timm_vision_transformer_large():
    model, example_inputs = torchbenchmark.models.timm_vision_transformer_large.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_pytorch_CycleGAN_and_pix2pix():
    model, example_inputs = torchbenchmark.models.pytorch_CycleGAN_and_pix2pix.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_dcgan():
    model, example_inputs = torchbenchmark.models.dcgan.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_vision_transformer():
    model, example_inputs = torchbenchmark.models.timm_vision_transformer.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_DALLE2_pytorch():
    model, example_inputs = torchbenchmark.canary_models.DALLE2_pytorch.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_pytorch_stargan():
    model, example_inputs = torchbenchmark.models.pytorch_stargan.Model(test="eval", device="cuda").get_module()
    model(*example_inputs)
