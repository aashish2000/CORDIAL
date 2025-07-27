import base64
from PIL import Image

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image_openai(image_input, image_mode, image_input_detail):
    if image_mode == 'url':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'path':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_input)}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {image_mode}.")
    
    return image_content

def process_image_qwen(image_input, image_mode, image_input_detail):
    if image_mode == 'url':
        image_content = {
            "type": "image_url",
            "image" : image_input,
        }
    elif image_mode == 'path':
        image_content = {
            "type": "image_url",
            "image_url" : f"data:image/jpeg;base64,{encode_image(image_input)}",
        }
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {image_mode}.")
    
    return image_content

def process_image_unsloth(image_input, image_mode, image_input_detail):
    # image_mode and image_input_detail are not used in this function
    return {
        "type": "image",
        "image": Image.open(image_input).convert("RGB"),
    }

def process_image_anthropic(image_input, image_mode, image_input_detail):
    # image_mode and image_input_detail are not used in this function
    image_content = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": encode_image(image_input),
        },
    }
    
    return image_content

def process_text(text_input):
    text_content = {
        "type": "text",
        "text": text_input,
    }

    return text_content

def build_sl_prompt(
    text_inputs, 
    image_inputs,
    caption,
    image,
    image_mode,
    image_input_detail,
    instruction,
    end_chat_with_assistant_msg = True,
    anthropic_inference = False,
    qwen_inference = False,
    unsloth_finetuning = False,
    alternate_user_assistant_msgs = False,
):
    if (anthropic_inference):
        process_image = process_image_anthropic
    elif (qwen_inference):
        process_image = process_image_qwen
    elif (unsloth_finetuning):
        process_image = process_image_unsloth
    else:
        process_image = process_image_openai

    messages = []

    if(instruction[0] != ""):
        messages.append({
            "role": "system",
            "content": [process_text(instruction[0])],
        })
    
    if_added_instr = False
    if (not alternate_user_assistant_msgs):
        messages.append({
            "role": "user",
            "content": [process_text(instruction[2])],
        })
        if_added_instr = True

    # Few-shot examples
    for i in range(len(text_inputs)):
        content = []
        if (i == 0 and alternate_user_assistant_msgs):
            content.append(process_text(instruction[2]))
            if_added_instr = True

        content.append(
            process_text(text_inputs[i].split("\n")[0])
        )
        messages.append({
            "role": "user",
            "content": content,
        })
        
        messages[-1]["content"].append(process_image(image_inputs[i], image_mode, image_input_detail))
        
        messages.append({
            "role": "assistant",
            "content": [process_text(text_inputs[i].split("\n")[1])],
        })

    # Image-caption pair
    if(image is not None and caption is not None):
        content = []
        if (not if_added_instr):
            content.append(process_text(instruction[2]))
            if_added_instr = True
        
        content.extend([
            process_text(
                "Caption: " + caption
            ),
            process_image(image, image_mode, image_input_detail)
        ])
        messages.append({
            "role": "user",
            "content": content,
        })
    
    if (not if_added_instr):
        messages.append({
            "role": "user",
            "content": [process_text(instruction[2])],
        })
        if_added_instr = True
    
    if (end_chat_with_assistant_msg):
        messages.append({
            "role": "assistant",
            "content": [process_text(instruction[3])],
        })
    
    return messages


def build_ml_prompt(
    text_inputs,
    image_inputs,
    caption,
    image,
    image_mode,
    image_input_detail,
    instruction,
    end_chat_with_assistant_msg = True,
    anthropic_inference = False,
    qwen_inference = False,
    unsloth_finetuning = False,
    alternate_user_assistant_msgs = False,
    qwen_train = False
):
    assert (len(text_inputs) == len(image_inputs))

    if (anthropic_inference):
        process_image = process_image_anthropic
    elif (qwen_inference):
        process_image = process_image_qwen
    elif (unsloth_finetuning):
        process_image = process_image_unsloth
    else:
        process_image = process_image_openai

    messages = []

    if (instruction[0] != ""):
        messages.append({
            "role": "system",
            "content": instruction[0]
        })
    
    if_added_instr = False
    if (not alternate_user_assistant_msgs):
        messages.append({
            "role": "user",
            "content": instruction[1],
        })
        if_added_instr = True

    if (len(text_inputs)):
        for text_input, image_input in zip(text_inputs, image_inputs):
            content = []
            if (not if_added_instr):
                content.append(process_text(instruction[1]))
                if_added_instr = True
            
            content.extend([
                process_text(
                    text_input.split('\n')[0]
                ),
                process_image(image_input, image_mode, image_input_detail)
            ])
            messages.append({
                "role": "user",
                "content": content,
            })
            messages.append({
                "role": "assistant",
                "content": text_input.split('\n')[1]
            })
    
    # Image-caption pair
    if (image is not None and caption is not None):
        content = []
        if (not if_added_instr):
            content.append(process_text(instruction[1]))
            if_added_instr = True
        content.extend([
            process_text("Caption: " + caption + '\n'),
            process_image(image, image_mode, image_input_detail)
        ])
        messages.append({
            "role": "user",
            "content": content
        })

    if (not if_added_instr):
        messages.append({
            "role": "user",
            "content": instruction[1]
        })
        if_added_instr = True

    if (end_chat_with_assistant_msg):
        messages.append({
            "role": "assistant",
            "content": instruction[2]
        })
    return messages