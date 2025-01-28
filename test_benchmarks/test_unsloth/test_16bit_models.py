from utils import load_and_prepare_model, run_inference

def test_deepseek_v3():
    model_name = "unsloth/DeepSeek-V3"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_gemma_2_9b():
    model_name = "unsloth/gemma-2-9b"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llama_2_7b_chat():
    model_name = "unsloth/llama-2-7b-chat"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llama_3_8b():
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llama_3_1_8b():
    model_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llama_3_1_8b_instruct():
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llama_3_2_3b():
    model_name = "unsloth/Llama-3.2-3B"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llama_3_2_3b_instruct():
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_llava_1_5_7b():
    model_name = "unsloth/llava-1.5-7b-hf"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_mistral_7b_instruct():
    model_name = "unsloth/mistral-7b-instruct-v0.3"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_phi_3_5_mini_instruct():
    model_name = "unsloth/Phi-3.5-mini-instruct"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_phi_3_mini_4k_instruct():
    model_name = "unsloth/Phi-3-mini-4k-instruct"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_phi_4():
    model_name = "unsloth/phi-4"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_qwen2_7b():
    model_name = "unsloth/Qwen2-7B"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_qwen2_vl_7b_instruct():
    model_name = "unsloth/Qwen2-VL-7B-Instruct"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")


def test_qwen2_5_3b():
    model_name = "unsloth/Qwen2.5-3B"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_smollm2_1_7B_instruct():
    model_name = "unsloth/SmolLM2-1.7B-Instruct"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_tinyllama_chat():
    model_name = "unsloth/tinyllama-chat"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")

def test_zephyr_sft():
    model_name = "unsloth/zephyr-sft"
    model, tokenizer = load_and_prepare_model(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    response = run_inference(model, tokenizer)
    print(f"--- Test: {model_name} ---")
    print(response)
    print("---------------------------------------")
