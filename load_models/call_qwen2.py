import os, sys
import torch
from typing import Literal
from helper import set_seed
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import INSTRUCTION_DEFAULT_SL_HARD, text_inputs_clue_sl_hard, image_inputs_clue_sl_hard
from environment import DATASET_ROOT_DIRS
from prompt_builder import build_sl_prompt, build_ml_prompt
from time import time

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_qwen2():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor

def call_qwen2(
    model,
    processor,
    label_type = Literal["multi_label", "single_label"],
    instruction = INSTRUCTION_DEFAULT_SL_HARD['gpt4o'],
    text_inputs = text_inputs_clue_sl_hard,
    image_inputs = image_inputs_clue_sl_hard,
    image = DATASET_ROOT_DIRS['clue_sl_hard']+"images/3661.jpg",
    caption = "all you need to know about tea",
    seed = 42,
    history = None,
    save_history = False,
):
    set_seed(seed)

    assert len(text_inputs) == len(image_inputs), "Number of text inputs and image inputs should be the same."
    assert (image is None and caption is None) or (image is not None and caption is not None), "Image and caption should be provided together."

    if label_type == "multi_label":
        prompt_image_eval = build_ml_prompt
    else:
        prompt_image_eval = build_sl_prompt

    prompt = prompt_image_eval(
        text_inputs,
        image_inputs,
        caption,
        image,
        image_mode = 'path',
        image_input_detail = 'high',
        instruction = instruction,
        qwen_inference = True,
    )

    if (history is not None):
        prompt = history + prompt

    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    output_dict = {}    
    time_start = time()

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    output_dict['description'] = output_text[0]
    
    prompt_history = prompt[:]
    
    if (label_type == 'single_label'):
        prompt_history[-1]['content'][0]['text'] += output_text[0]
    else:
        prompt_history[-1]['content'] += output_text[0]

    if save_history: output_dict['history'] = prompt_history

    time_end = time()
    output_dict['time'] = time_end - time_start

    return output_dict