import torchbenchmark.models.Background_Matting
import torchbenchmark.models.Super_SloMo

def test_Background_Matting():
    model, example_inputs = torchbenchmark.models.Background_Matting.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_Super_SloMo():
    model, example_inputs = torchbenchmark.models.Super_SloMo.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
