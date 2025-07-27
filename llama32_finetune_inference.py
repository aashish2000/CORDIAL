import os
import argparse
from typing import Literal
from helper import read_json, read_csv, write_csv, clean_text
from prompt_builder import build_sl_prompt, build_ml_prompt
from configs import test_metadata_paths, instruction_default_mapping, text_examples_mapping, image_examples_mapping
from environment import DATASET_ROOT_DIRS
from unsloth import FastVisionModel

def eval_custom_dataset_sl(
    model,
    tokenizer,
    metadata_path, 
    root_dir, 
    instruction,
    text_examples,
    image_examples,
    results_file,
    shot: Literal['zero', 'few'],
    max_new_tokens: int,
    do_sample: bool
):
    metadata = read_json(metadata_path)
    results = read_csv(results_file) if os.path.exists(results_file) else [['Image', 'Caption', 'True Labels', 'Predicted Labels']]
    
    total = len(metadata)
    for i, item in enumerate(metadata):
        print(f"\nProcessing {i+1}/{total}")
        
        if (i + 1) < len(results):
            continue

        caption, filename, label = item["caption"], item["filename"], item['labels'][0]
        image_path = os.path.join(root_dir, filename)
        
        text_examples_input = text_examples if shot == 'few' else []
        image_examples_input = image_examples if shot == 'few' else []

        messages = build_sl_prompt(
            text_inputs=text_examples_input, image_inputs=image_examples_input,
            caption=caption, image=image_path, image_mode='path', image_input_detail='high',
            instruction=instruction, unsloth_finetuning=True, end_chat_with_assistant_msg=False
        )
        messages[0]['role'] = 'user' # Ensure consistency

        images = []
        for msg in messages:
            for content_item in msg['content']:
                if 'image' in content_item:
                    images.append(content_item.pop('image'))
        
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(images, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        result_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = result_tokens[:, prompt_length:]
        predicted_label = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        predicted_label = clean_text(predicted_label)

        print(f"Pred: {predicted_label} | True: {label}")
        results.append([filename, caption, label, predicted_label])
        write_csv(results_file, results)

def eval_custom_dataset_ml(
    model,
    tokenizer,
    metadata_path, 
    root_dir, 
    instruction,
    text_examples,
    image_examples,
    results_file,
    shot: Literal['zero', 'few'],
    max_new_tokens: int,
    do_sample: bool
):
    metadata = read_json(metadata_path)
    results = read_csv(results_file) if os.path.exists(results_file) else [['Image', 'Caption', 'True Labels', 'Predicted Labels']]
    
    total = len(metadata)
    for i, item in enumerate(metadata):
        print(f"\nProcessing {i+1}/{total}")

        if (i + 1) < len(results):
            continue

        caption, filename, labels = item["caption"], item["filename"], item['labels']
        image_path = os.path.join(root_dir, filename)

        text_examples_input = text_examples if shot == 'few' else []
        image_examples_input = image_examples if shot == 'few' else []

        messages = build_ml_prompt(
            text_inputs=text_examples_input, image_inputs=image_examples_input,
            caption=caption, image=image_path, image_mode='path', image_input_detail='high',
            instruction=instruction, unsloth_finetuning=True, end_chat_with_assistant_msg=False
        )
        messages[0]['role'] = 'user'

        images = []
        for msg in messages:
            for content_item in msg['content']:
                if 'image' in content_item:
                    images.append(content_item.pop('image'))

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(images, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

        result_tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = result_tokens[:, prompt_length:]
        predicted_labels_raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        
        ref_labels = ['Action', 'Meta', 'Story', 'Subjective', 'Visible']
        csv_list = predicted_labels_raw.split(',')
        predicted_labels = [clean_text(x.strip()).title() for x in csv_list]
        predicted_labels = sorted(list(dict.fromkeys(predicted_labels)))
        predicted_labels = [label for label in predicted_labels if label in ref_labels]

        print(f"Pred: {predicted_labels} | True: {labels}")
        results.append([filename, caption, str(labels), str(predicted_labels)])
        write_csv(results_file, results)

def main(args):
    """Main function to set up and run the evaluation."""
    # --- 1. Load Model and Tokenizer ---
    print(f"🚀 Loading model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    print("✅ Model loaded successfully.")

    # --- 2. Get Dataset-specific Configs ---
    dataset_name = args.dataset_name
    try:
        instruction = instruction_default_mapping[dataset_name]['ft_llama32']
    except KeyError:
        instruction = instruction_default_mapping[dataset_name]['gpt4o']
    
    text_examples = text_examples_mapping.get(dataset_name, [])
    image_examples = image_examples_mapping.get(dataset_name, [])
    metadata_path = test_metadata_paths[dataset_name]
    root_dir = DATASET_ROOT_DIRS[dataset_name]

    # --- 3. Run Evaluation based on dataset name ---
    if args.dataset_name == 'clue_ml':
        print(f"\n🔬 Detected '{args.dataset_name}'. Using multi-label evaluation ({args.shot}-shot)...")
        eval_function = eval_custom_dataset_ml
    else:
        print(f"\n🔬 Using default single-label evaluation for '{args.dataset_name}' ({args.shot}-shot)...")
        eval_function = eval_custom_dataset_sl
    
    eval_function(
        model=model, tokenizer=tokenizer,
        metadata_path=metadata_path, root_dir=root_dir, instruction=instruction,
        text_examples=text_examples, image_examples=image_examples,
        results_file=args.results_file, shot=args.shot,
        max_new_tokens=args.max_new_tokens, do_sample=args.do_sample
    )
    print(f"\n🎉 Evaluation complete. Results saved to {args.results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a finetuned Vision Language Model on CORDIAL")

    # --- Required Arguments ---
    parser.add_argument('--model_name', type=str, required=True, help='Path to the finetuned model directory.')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', required=True, 
                        choices=['clue_ml', 'clue_sl_hard', 'disrel_sl', 'tweets_sl'])
    parser.add_argument('--results_file', type=str, required=True, help='Path to the output CSV file for saving results.')

    # --- Optional Arguments ---
    parser.add_argument('--shot', type=str, default='zero', choices=['zero', 'few'], help='Prompting strategy: zero-shot or few-shot. (default: zero)')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load the model in 4-bit precision.')
    parser.add_argument('--do_sample', action='store_true', help='Enable sampling during generation (default: False).')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate. (default: 512)')

    args = parser.parse_args()
    main(args)