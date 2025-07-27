import os, sys
import torch
from typing import Literal
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from helper import set_seed
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

from configs import INSTRUCTION_DEFAULT_SL_HARD, text_inputs_clue_sl_hard, image_inputs_clue_sl_hard
from environment import DATASET_ROOT_DIRS
from prompt_builder import build_sl_prompt, build_ml_prompt
from time import time


def load_llava16(context_len = 32768):
    # Configure LlaVA
    model_path = "liuhaotian/llava-v1.6-34b"

    pipe = pipeline(
        model_path, 
        TurbomindEngineConfig(
            session_len=context_len,
            tp=torch.cuda.device_count()
        ), 
    )
    print(f'{model_path} loaded.')
    return pipe

def call_llava16(
    llava_model,
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
        end_chat_with_assistant_msg = False
    )

    output_dict = {}    
    llava_start = time()

    sess = llava_model.chat(
        prompt,
        gen_config=GenerationConfig(
            temperature=0,
            max_new_tokens=512,
            random_seed=seed
        ),
        session=history
    )
    output_dict['description'] = sess.response.text
    
    if save_history: output_dict['history'] = sess
    llava_end = time()
    output_dict['time'] = llava_end - llava_start

    return output_dict