import torchbenchmark.models.speech_transformer
import torchbenchmark.models.demucs
import torchbenchmark.models.tacotron2
import torchbenchmark.models.tts_angular

def test_speech_transformer():
    model, example_inputs = torchbenchmark.models.speech_transformer.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_demucs():
    model, example_inputs = torchbenchmark.models.demucs.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_tacotron2():
    model, example_inputs = torchbenchmark.models.tacotron2.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_tts_angular():
    model, example_inputs = torchbenchmark.models.tts_angular.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
