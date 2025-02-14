#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
New models after the paper TorchBench
"""

# ------------------------------------------------
# 1. Recommendation Systems
# Models: dlrm
import torchbenchmark.models.dlrm

def test_dlrm():
    model, example_inputs = torchbenchmark.models.dlrm.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)


# ------------------------------------------------
# 2. Image Generation
# Models: stable_diffusion_text_encoder, stable_diffusion_unet
import torchbenchmark.models.stable_diffusion_text_encoder
import torchbenchmark.models.stable_diffusion_unet

def test_stable_diffusion_text_encoder():
    model, example_inputs = torchbenchmark.models.stable_diffusion_text_encoder.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_stable_diffusion_unet():
    model, example_inputs = torchbenchmark.models.stable_diffusion_unet.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)


# ------------------------------------------------
# 3. LLM / Multimodal Models
# Models:
#   - hf_GPT2, hf_GPT2_large, hf_Roberta_base
#   - hf_Whisper, hf_distil_whisper
#   - llama, llama_v2_7b_16h, llava
#   - moondream, torch_multimodal_clip
import torchbenchmark.models.hf_GPT2
import torchbenchmark.models.hf_GPT2_large
import torchbenchmark.models.hf_Roberta_base
import torchbenchmark.models.hf_Whisper
import torchbenchmark.models.hf_distil_whisper
import torchbenchmark.models.llama
import torchbenchmark.models.llama_v2_7b_16h
import torchbenchmark.models.llava
import torchbenchmark.models.moondream
# import torchbenchmark.models.torch_multimodal_clip

def test_hf_GPT2():
    model, example_inputs = torchbenchmark.models.hf_GPT2.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_GPT2_large():
    model, example_inputs = torchbenchmark.models.hf_GPT2_large.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Roberta_base():
    model, example_inputs = torchbenchmark.models.hf_Roberta_base.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_Whisper():
    model, example_inputs = torchbenchmark.models.hf_Whisper.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_hf_distil_whisper():
    model, example_inputs = torchbenchmark.models.hf_distil_whisper.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_llama():
    model, example_inputs = torchbenchmark.models.llama.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_llama_v2_7b_16h():
    model, example_inputs = torchbenchmark.models.llama_v2_7b_16h.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_llava():
    model, example_inputs = torchbenchmark.models.llava.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

def test_moondream():
    model, example_inputs = torchbenchmark.models.moondream.Model(test="eval", device="cuda", batch_size=1).get_module()
    model(*example_inputs)

# def test_torch_multimodal_clip():
#     model, example_inputs = torchbenchmark.models.torch_multimodal_clip.Model(test="eval", device="cuda", batch_size=1).get_module()
#     model(*example_inputs)
