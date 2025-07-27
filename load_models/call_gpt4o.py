from environment import DATASET_ROOT_DIRS, OPENAI_API_KEY
from configs import INSTRUCTION_DEFAULT_SL_HARD, text_inputs_clue_sl_hard, image_inputs_clue_sl_hard
from prompt_builder import build_sl_prompt, build_ml_prompt
from openai import OpenAI
import os, requests, sys
from typing import Literal
from time import time

client = OpenAI()
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from helper import retry_if_fail

@retry_if_fail
def call_gpt4o(
    image_mode: Literal['url', 'path'] = 'path',
    image_input_detail: Literal['low', 'high'] = 'low',
    max_tokens = 512,
    seed = 42,
    instruction = INSTRUCTION_DEFAULT_SL_HARD['gpt4o'],
    label_type = Literal["multi_label", "single_label"],
    text_inputs = text_inputs_clue_sl_hard,
    image_inputs = image_inputs_clue_sl_hard,
    image = DATASET_ROOT_DIRS['clue_sl_hard']+"images/3661.jpg",
    caption = "all you need to know about tea",
    history = None,
    save_history = False,
):

    if label_type == "multi_label":
        prompt_image_eval = build_ml_prompt
    else:
        prompt_image_eval = build_sl_prompt

    messages = prompt_image_eval(
        text_inputs, 
        image_inputs, 
        caption,
        image,
        image_mode,
        image_input_detail,
        instruction,
    )

    if history is not None: messages = history + messages
    
    output_dict = {}
    
    payload = {
        'model':"gpt-4o",
        'messages':messages,
        'max_tokens':max_tokens,
        'seed': seed,
    }
    
    gpt4o_start = time()
    if image_mode == 'url':
        response = client.chat.completions.create(**payload)
        output_dict['description'] = response.choices[0].message.content
    elif image_mode == 'path':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload,
        )
        output_dict['description'] = response.json()['choices'][0]['message']['content']
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {mode}.")    
    
    gpt4o_end = time()
    output_dict['time'] = gpt4o_end - gpt4o_start
    
    if save_history:
        output_dict['history'] = messages + [{
            'role': 'assistant', 
            'content': output_dict['description'],
        }]
    
    return output_dict
