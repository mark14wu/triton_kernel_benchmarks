from unsloth import FastLanguageModel

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def load_and_prepare_model(model_name, max_seq_length=4096, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def run_inference(model, tokenizer, max_new_tokens=100):
    instruction = "You are a helpful assistant who can answer questions"
    input_text = "Who developed GPT models?"

    inputs = tokenizer([alpaca_prompt.format(instruction, input_text, "")], return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(outputs)[0]
    return response