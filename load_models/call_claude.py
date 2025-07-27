import os, sys
import google.auth
import google.auth.transport.requests
from anthropic import AnthropicVertex
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import INSTRUCTION_DEFAULT_SL_HARD, text_inputs_clue_sl_hard, image_inputs_clue_sl_hard
from environment import DATASET_ROOT_DIRS, LOCATION
from prompt_builder import build_sl_prompt, build_ml_prompt
import time
from helper import retry_if_fail
from typing import Literal

def refresh_creds(creds):
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

# Setup client with Google Cloud credentials
def load_claude():
    creds, project_id = google.auth.default()
    refresh_creds(creds)

    client = AnthropicVertex(
        project_id=project_id, 
        region=LOCATION,
        credentials=creds
    )
    return client, creds

@retry_if_fail
def call_claude(
    client,
    creds,
    label_type = Literal["multi_label", "single_label"],
    instruction = INSTRUCTION_DEFAULT_SL_HARD['gpt4o'],
    text_inputs = text_inputs_clue_sl_hard,
    image_inputs = image_inputs_clue_sl_hard,
    image = DATASET_ROOT_DIRS['clue_sl_hard']+"images/3661.jpg",
    caption = "all you need to know about tea",
    seed = 42,  # cannot set seed for Claude, but keeping it for consistency
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
        instruction = instruction,
        anthropic_inference = True,
        end_chat_with_assistant_msg = False
    )

    # Remove system message from prompt, to pass in as a separate argument
    system_msg = None
    if prompt[0]['role'] == 'system': 
        if (label_type == 'single_label'):
            system_msg = prompt[0]['content'][0]['text']
            prompt = prompt[1:]
        else:
            system_msg = prompt[0]['content']
            prompt = prompt[1:]

    if history is not None: prompt = history + prompt

    output_dict = {}
    if save_history: output_dict = {'history': prompt}

    claude_start = time.time()

    if (creds.expired):
        refresh_creds(creds)
        client.api_key = creds.token

    if (system_msg):
        response = client.messages.create(
            model = 'claude-3-5-sonnet-v2@20241022',
            messages = prompt,
            system = system_msg,
            max_tokens = 512,
        )
    else:
        response = client.messages.create(
            model = 'claude-3-5-sonnet-v2@20241022',
            messages = prompt,
            max_tokens = 512,
        )

    output_dict['description'] = response.content[0].text

    claude_end = time.time()
    output_dict['time'] = claude_end - claude_start

    if save_history: 
        output_dict['history'] = prompt + [
            {
                'role': 'assistant',
                'content' : output_dict['description']
            }
        ]
    
    return output_dict