import torchbenchmark.models.detectron2_fasterrcnn_r_101_c4
import torchbenchmark.models.detectron2_fasterrcnn_r_50_dc5
import torchbenchmark.models.doctr_reco_predictor
import torchbenchmark.models.detectron2_fasterrcnn_r_101_dc5
import torchbenchmark.models.detectron2_fasterrcnn_r_50_fpn
import torchbenchmark.models.timm_efficientdet
import torchbenchmark.models.detectron2_fasterrcnn_r_101_fpn
import torchbenchmark.models.detectron2_maskrcnn
import torchbenchmark.models.timm_vovnet
import torchbenchmark.models.detectron2_fasterrcnn_r_50_c4
import torchbenchmark.models.doctr_det_predictor
import torchbenchmark.models.vision_maskrcnn

def test_detectron2_fasterrcnn_r_101_c4():
    model, example_inputs = torchbenchmark.models.detectron2_fasterrcnn_r_101_c4.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_fasterrcnn_r_50_dc5():
    model, example_inputs = torchbenchmark.models.detectron2_fasterrcnn_r_50_dc5.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_doctr_reco_predictor():
    model, example_inputs = torchbenchmark.models.doctr_reco_predictor.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_fasterrcnn_r_101_dc5():
    model, example_inputs = torchbenchmark.models.detectron2_fasterrcnn_r_101_dc5.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_fasterrcnn_r_50_fpn():
    model, example_inputs = torchbenchmark.models.detectron2_fasterrcnn_r_50_fpn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_efficientdet():
    model, example_inputs = torchbenchmark.models.timm_efficientdet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_fasterrcnn_r_101_fpn():
    model, example_inputs = torchbenchmark.models.detectron2_fasterrcnn_r_101_fpn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_maskrcnn():
    model, example_inputs = torchbenchmark.models.detectron2_maskrcnn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_timm_vovnet():
    model, example_inputs = torchbenchmark.models.timm_vovnet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_detectron2_fasterrcnn_r_50_c4():
    model, example_inputs = torchbenchmark.models.detectron2_fasterrcnn_r_50_c4.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_doctr_det_predictor():
    model, example_inputs = torchbenchmark.models.doctr_det_predictor.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_vision_maskrcnn():
    model, example_inputs = torchbenchmark.models.vision_maskrcnn.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
