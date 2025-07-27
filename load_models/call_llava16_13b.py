import os, sys
from typing import Literal
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import openai

from configs import INSTRUCTION_DEFAULT_SL_HARD, text_inputs_clue_sl_hard, image_inputs_clue_sl_hard
from environment import DATASET_ROOT_DIRS
from prompt_builder import build_sl_prompt, build_ml_prompt
from time import time

def load_llava16_13b():
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="1234",
    )
    return client

def call_llava16_13b(
    client: openai.OpenAI,
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
        alternate_user_assistant_msgs=True,
        end_chat_with_assistant_msg=False,
    )

    if (history is not None):
        prompt = history + prompt

    output_dict = {}    
    time_start = time()

    response = client.chat.completions.create(
        model="llava-hf/llava-v1.6-vicuna-13b-hf",
        messages=prompt,
        seed=seed,
        max_tokens=512,
        temperature=0,
    )
    output_dict['description'] = response.choices[0].message.content

    prompt_history = prompt[:]
    
    prompt_history.append({
        "role": "assistant",
        "content": output_dict['description']
    })
    
    if save_history: output_dict['history'] = prompt_history
    time_end = time()
    output_dict['time'] = time_end - time_start

    return output_dict