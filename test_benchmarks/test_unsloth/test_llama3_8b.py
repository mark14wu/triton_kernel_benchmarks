from pyexpat import model
from unsloth import FastLanguageModel
import torch

def _prepare_model():
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def test_sample_prompts():
    model, tokenizer = _prepare_model()
    instruction = "You are a helpful assistant who can answer questions"
    input_text = "Who developed GPT models?"

    inputs = tokenizer([alpaca_prompt.format(instruction, input_text, "")], return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.batch_decode(outputs)[0]
    print(response)
