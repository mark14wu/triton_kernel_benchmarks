from pdb import run
from pyexpat import model
from utils import load_and_prepare_model, run_inference

def run_model_unittests(model_name):
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

def test_deepseek_v3_gguf():
    model_name = "unsloth/DeepSeek-V3-GGUF"
    run_model_unittests(model_name)

def test_gemma_2_9b_bnb_4bit():
    model_name = "unsloth/gemma-2-9b-bnb-4bit"
    run_model_unittests(model_name)

def test_gemma_2_9b_instruct_bnb_4bit():
    model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_2_7b_bnb_4bit():
    model_name = "unsloth/llama-2-7b-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_2_7b_instruct_bnb_4bit():
    model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_8b_bnb_4bit():
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_8b_instruct_bnb_4bit():
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_1_8b_bnb_4bit():
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_1_8b_instruct_bnb_4bit():
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_2_3b_bnb_4bit():
    model_name = "unsloth/Llama-3.2-3B-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_2_3b_instruct_bnb_4bit():
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_llama_3_2_3b_instruct_gguf():
    model_name = "unsloth/Llama-3.2-3B-Instruct-GGUF"
    run_model_unittests(model_name)

def test_llava_1_5_7b_hf_bnb_4bit():
    model_name = "unsloth/llava-1.5-7b-hf-bnb-4bit"
    run_model_unittests(model_name)

def test_llava_1_6_mistral_7b_hf_bnb_4bit():
    model_name = "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
    run_model_unittests(model_name)

def test_mistral_7b_0_2_bnb_4bit():
    model_name = "unsloth/mistral-7b-v0.2-bnb-4bit"
    run_model_unittests(model_name)

def test_mistral_7b_0_3_bnb_4bit():
    model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    run_model_unittests(model_name)

def test_mistral_7b_instruct_0_2_bnb_4bit():
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    run_model_unittests(model_name)

def test_mistral_7b_instrcut_0_3_bnb_4bit():
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    run_model_unittests(model_name)

def test_phi_3_mini_4k_instruct_bnb_4bit():
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_phi_3_medium_4k_instruct_bnb_4bit():
    model_name = "unsloth/Phi-3-medium-4k-instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_phi_3_5_mini_instruct_bnb_4bit():
    model_name = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_phi_4_bnb_4bit():
    model_name = "unsloth/phi-4-bnb-4bit"
    run_model_unittests(model_name)

def test_phi_4_bnb_4bit_dynamic():
    model_name = "unsloth/phi-4-unsloth-bnb-4bit"
    run_model_unittests(model_name)

def test_phi_4_gguf():
    model_name = "unsloth/phi-4-GGUF"
    run_model_unittests(model_name)

def test_qwen2_7b_bnb_4bit():
    model_name = "unsloth/Qwen2-7B-bnb-4bit"
    run_model_unittests(model_name)

def test_qwen2_7b_instruct_bnb_4bit():
    model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_qwen2_vl_7b_instruct_unsloth_bnb_4bit():
    model_name = "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit"
    run_model_unittests(model_name)

def test_qwen_2_5_7b_bnb_4bit():
    model_name = "unsloth/Qwen2.5-7B-bnb-4bit"
    run_model_unittests(model_name)

def test_qwen_2_5_7b_instruct_bnb_4bit():
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_smollm2_1_7b_bnb_4bit():
    model_name = "unsloth/SmolLM2-1.7B-bnb-4bit"
    run_model_unittests(model_name)

def test_smollm2_1_7b_instruct_bnb_4bit():
    model_name = "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit"
    run_model_unittests(model_name)

def test_tinyllama_bnb_4bit():
    model_name = "unsloth/tinyllama-bnb-4bit"
    run_model_unittests(model_name)

def test_tinyllama_chat_bnb_4bit():
    model_name = "unsloth/tinyllama-chat-bnb-4bit"
    run_model_unittests(model_name)

def test_zephyr_sft_bnb_4bit():
    model_name = "unsloth/zephyr-sft-bnb-4bit"
    run_model_unittests(model_name)

def test_codellama_7b_bnb_4bit():
    model_name = "unsloth/codellama-7b-bnb-4bit"
    run_model_unittests(model_name)

def test_yi_6b_bnb_4bit():
    model_name = "unsloth/yi-6b-bnb-4bit"
    run_model_unittests(model_name)