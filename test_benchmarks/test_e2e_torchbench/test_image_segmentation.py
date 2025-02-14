import torchbenchmark.models.detectron2_maskrcnn_r_101_c4 as dmrcnn101_c4
import torchbenchmark.models.detectron2_maskrcnn_r_50_fpn as dmrcnn50_fpn
import torchbenchmark.models.yolov3
import torchbenchmark.models.detectron2_maskrcnn_r_101_fpn as dmrcnn101_fpn
import torchbenchmark.models.detectron2_fcos_r_50_fpn
import torchbenchmark.models.detectron2_maskrcnn_r_50_c4 as dmrcnn50_c4
import torchbenchmark.models.pytorch_unet
import torchbenchmark.models.sam
import torchbenchmark.models.sam_fast

def test_detectron2_maskrcnn_r_101_c4_seg():
    model, example_inputs = dmrcnn101_c4.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_maskrcnn_r_50_fpn_seg():
    model, example_inputs = dmrcnn50_fpn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_yolov3():
    model, example_inputs = torchbenchmark.models.yolov3.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_maskrcnn_r_101_fpn_seg():
    model, example_inputs = dmrcnn101_fpn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_fcos_r_50_fpn():
    model, example_inputs = torchbenchmark.models.detectron2_fcos_r_50_fpn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_maskrcnn_r_50_c4_seg():
    model, example_inputs = dmrcnn50_c4.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_pytorch_unet():
    model, example_inputs = torchbenchmark.models.pytorch_unet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_sam():
    model, example_inputs = torchbenchmark.models.sam.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_sam_fast():
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model, example_inputs = torchbenchmark.models.sam_fast.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
