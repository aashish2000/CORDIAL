import os, sys
import google.auth
import google.auth.transport.requests
import openai
from typing import Literal
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import INSTRUCTION_DEFAULT_SL_HARD, text_inputs_clue_sl_hard, image_inputs_clue_sl_hard
from environment import DATASET_ROOT_DIRS, LOCATION
from prompt_builder import build_sl_prompt, build_ml_prompt
from time import time

def refresh_creds(creds):
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

# Setup client with Google Cloud credentials
def load_gemini_flash():
    creds, project_id = google.auth.default()
    refresh_creds(creds)

    client = openai.OpenAI(
        base_url=f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{LOCATION}/endpoints/openapi",
        api_key=creds.token
    )
    return client, creds

def call_gemini_flash(
    client,
    creds,
    instruction = INSTRUCTION_DEFAULT_SL_HARD['gpt4o'],
    label_type = Literal["multi_label", "single_label"],
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
        image_input_detail = 'low',
        instruction = instruction
    )

    if history is not None: prompt = history + prompt

    output_dict = {}
    if save_history: output_dict = {'history': prompt}

    gemini_start = time()

    if (creds.expired):
        refresh_creds(creds)
        client.api_key = creds.token

    response = client.chat.completions.create(
        model = 'google/gemini-1.5-flash-002',
        messages = prompt,
        seed = seed
    )
    output_dict['description'] = response.choices[0].message.content

    gemini_end = time()
    output_dict['time'] = gemini_end - gemini_start

    if save_history:        
        output_dict['history'] = prompt + [
            {
                'role': 'assistant',
                'content' : output_dict['description']
            }
        ]
    
    return output_dict