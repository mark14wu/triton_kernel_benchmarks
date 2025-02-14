import torchbenchmark.models.BERT_pytorch
import torchbenchmark.models.hf_Bert_large
import torchbenchmark.models.hf_Longformer
# import torchbenchmark.canary_models.fambench_xlmr
import torchbenchmark.models.hf_BigBird
import torchbenchmark.models.hf_Reformer
import torchbenchmark.models.hf_Albert
import torchbenchmark.models.hf_DistilBert
import torchbenchmark.models.hf_T5
import torchbenchmark.models.hf_Bart
# import torchbenchmark.models.hf_public_text_generator1
import torchbenchmark.models.hf_T5_base
import torchbenchmark.models.hf_Bert
# import torchbenchmark.models.hf_public_text_generator1_large
import torchbenchmark.models.hf_T5_large
# import torchbenchmark.models.attention_is_all_you_need_pytorch
import torchbenchmark.models.fastNLP_Bert

def test_BERT_pytorch():
    model, example_inputs = torchbenchmark.models.BERT_pytorch.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Bert_large():
    model, example_inputs = torchbenchmark.models.hf_Bert_large.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Longformer():
    model, example_inputs = torchbenchmark.models.hf_Longformer.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

# def test_fambench_xlmr():
#     model, example_inputs = torchbenchmark.models.fambench_xlmr.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)

def test_hf_BigBird():
    model, example_inputs = torchbenchmark.models.hf_BigBird.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Reformer():
    model, example_inputs = torchbenchmark.models.hf_Reformer.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Albert():
    model, example_inputs = torchbenchmark.models.hf_Albert.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_DistilBert():
    model, example_inputs = torchbenchmark.models.hf_DistilBert.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_T5():
    model, example_inputs = torchbenchmark.models.hf_T5.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Bart():
    model, example_inputs = torchbenchmark.models.hf_Bart.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

# def test_hf_public_text_generator1():
#     model, example_inputs = torchbenchmark.models.hf_public_text_generator1.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)

def test_hf_T5_base():
    model, example_inputs = torchbenchmark.models.hf_T5_base.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Bert():
    model, example_inputs = torchbenchmark.models.hf_Bert.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

# def test_hf_public_text_generator1_large():
#     model, example_inputs = torchbenchmark.models.hf_public_text_generator1_large.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)

def test_hf_T5_large():
    model, example_inputs = torchbenchmark.models.hf_T5_large.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

# def test_attention_is_all_you_need_pytorch():
#     model, example_inputs = torchbenchmark.models.attention_is_all_you_need_pytorch.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)

def test_fastNLP_Bert():
    model, example_inputs = torchbenchmark.models.fastNLP_Bert.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)
