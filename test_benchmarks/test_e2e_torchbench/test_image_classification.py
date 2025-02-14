import torchbenchmark.models.alexnet
import torchbenchmark.models.phlippe_resnet
import torchbenchmark.models.squeezenet1_1
import torchbenchmark.models.densenet121
import torchbenchmark.models.resnet152
import torchbenchmark.models.mnasnet1_0
import torchbenchmark.models.resnet18
import torchbenchmark.models.mobilenet_v2
import torchbenchmark.models.resnet50
import torchbenchmark.models.mobilenet_v2_quantized_qat
import torchbenchmark.models.resnet50_quantized_qat
import torchbenchmark.models.mobilenet_v3_large
import torchbenchmark.models.resnext50_32x4d
import torchbenchmark.models.vgg16
import torchbenchmark.models.phlippe_densenet
import torchbenchmark.models.shufflenet_v2_x1_0
import torchbenchmark.models.timm_efficientnet
import torchbenchmark.models.timm_nfnet
import torchbenchmark.models.timm_regnet
import torchbenchmark.models.timm_resnest

def test_alexnet():
    model, example_inputs = torchbenchmark.models.alexnet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_phlippe_resnet():
    model, example_inputs = torchbenchmark.models.phlippe_resnet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_squeezenet1_1():
    model, example_inputs = torchbenchmark.models.squeezenet1_1.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_densenet121():
    model, example_inputs = torchbenchmark.models.densenet121.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_resnet152():
    model, example_inputs = torchbenchmark.models.resnet152.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_mnasnet1_0():
    model, example_inputs = torchbenchmark.models.mnasnet1_0.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_resnet18():
    model, example_inputs = torchbenchmark.models.resnet18.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_mobilenet_v2():
    model, example_inputs = torchbenchmark.models.mobilenet_v2.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_resnet50():
    model, example_inputs = torchbenchmark.models.resnet50.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_mobilenet_v2_quantized_qat():
    model, example_inputs = torchbenchmark.models.mobilenet_v2_quantized_qat.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_resnet50_quantized_qat():
    model, example_inputs = torchbenchmark.models.resnet50_quantized_qat.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_mobilenet_v3_large():
    model, example_inputs = torchbenchmark.models.mobilenet_v3_large.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_resnext50_32x4d():
    model, example_inputs = torchbenchmark.models.resnext50_32x4d.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_vgg16():
    model, example_inputs = torchbenchmark.models.vgg16.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_phlippe_densenet():
    model, example_inputs = torchbenchmark.models.phlippe_densenet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_shufflenet_v2_x1_0():
    model, example_inputs = torchbenchmark.models.shufflenet_v2_x1_0.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_efficientnet():
    model, example_inputs = torchbenchmark.models.timm_efficientnet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_nfnet():
    model, example_inputs = torchbenchmark.models.timm_nfnet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_regnet():
    model, example_inputs = torchbenchmark.models.timm_regnet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_resnest():
    model, example_inputs = torchbenchmark.models.timm_resnest.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
