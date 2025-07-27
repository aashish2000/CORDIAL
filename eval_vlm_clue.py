from environment import DATASET_ROOT_DIRS

# Import dataset-specific metadata paths, instructions and examples
from configs import test_metadata_paths
from configs import instruction_default_mapping, instruction_cot_mapping
from configs import text_examples_mapping, image_examples_mapping

# Import CLUE prompts and examples for default values
from configs import INSTRUCTION_DEFAULT_SL_HARD, INSTRUCTION_COT_SL_HARD, INSTRUCTION_DEFAULT_ML, INSTRUCTION_COT_ML
from configs import text_inputs_clue_sl_hard, image_inputs_clue_sl_hard, text_inputs_clue_ml, text_inputs_clue_ml_csv, image_inputs_clue_ml

from helper import write_csv, read_csv, read_json, clean_text, clean_json_string
import json
import os
import argparse

CSV_OUTPUT_MODELS = ["llama32", "qwen2", "phi35", "llava_ov", "llava16_13b"]

def load_model(model):
    if (model == 'llava16'):
        from load_models.call_llava16 import load_llava16, call_llava16
        model = load_llava16()
        return lambda configs: call_llava16(
            model,
            **configs
        )
    if (model == 'llava16_13b'):
        from load_models.call_llava16_13b import load_llava16_13b, call_llava16_13b
        model = load_llava16_13b()
        return lambda configs: call_llava16_13b(
            model,
            **configs
        )
    if (model == 'llava16_7b'):
        from load_models.call_llava16_7b import load_llava16_7b, call_llava16_7b
        model = load_llava16_7b()
        return lambda configs: call_llava16_7b(
            model,
            **configs
        )
    if (model == 'gemini'):
        from load_models.call_gemini import load_gemini, call_gemini
        client, creds = load_gemini()
        return lambda configs: call_gemini(
            client,
            creds,
            **configs
        )
    if (model == 'gpt4o'):
        from load_models.call_gpt4o import call_gpt4o
        return lambda configs: call_gpt4o(
            **configs,
        )
    if (model == 'claude'):
        from load_models.call_claude import load_claude, call_claude
        client, creds = load_claude()
        return lambda configs: call_claude(
            client,
            creds,
            **configs
        )
    if (model == 'qwen2'):
        from load_models.call_qwen2 import load_qwen2, call_qwen2
        model, processor = load_qwen2()
        return lambda configs: call_qwen2(
            model,
            processor,
            **configs
        )
    if (model == 'llama32'):
        from load_models.call_llama32 import load_llama32, call_llama32
        model = load_llama32()
        return lambda configs: call_llama32(
            model,
            **configs
        )
    if (model == 'internvl25'):
        from load_models.call_internvl25 import load_internvl25, call_internvl25
        model = load_internvl25()
        return lambda configs: call_internvl25(
            model,
            **configs
        )
    if (model == 'phi35'):
        from load_models.call_phi35 import load_phi35, call_phi35
        model = load_phi35()
        return lambda configs: call_phi35(
            model,
            **configs
        )
    if (model == 'llava_ov'):
        from load_models.call_llava_ov import load_llava_ov, call_llava_ov
        model = load_llava_ov()
        return lambda configs: call_llava_ov(
            model,
            **configs
        )
    if (model == 'gemini_flash'):
        from load_models.call_gemini_flash import load_gemini_flash, call_gemini_flash
        client, creds = load_gemini_flash()
        return lambda configs: call_gemini_flash(
            client,
            creds,
            **configs
        )
    raise NotImplementedError(f"Model {model} not implemented!")

def eval_model_default_sl(
    model,
    dataset,
    text_inputs = text_inputs_clue_sl_hard,
    image_inputs = image_inputs_clue_sl_hard,
    instruction = INSTRUCTION_DEFAULT_SL_HARD['gpt4o'],
    shot = 'zero',
    metadata_path = "metadata.json",
    results_file = './outputs/results_default.csv',
    seed = 42
):
    data = read_json(metadata_path)
    call_model = load_model(model)

    if(os.path.exists(results_file)):
        results = read_csv(results_file)
    else:
        results = [['Image', 'Caption', 'True Labels', 'Predicted Labels']]

    total = len(data)
    for i, item in enumerate(data):
        print()
        print(f"{i+1}/{total}")

        caption = item["caption"]
        filename = item["filename"]

        label = item['labels'][0]

        text_inputs_input = [] if shot == 'zero' else text_inputs
        image_inputs_input = [] if shot == 'zero' else image_inputs

        if (i+1 < len(results)):
            continue

        predicted_label = call_model({
            'label_type': 'single_label',
            'image': DATASET_ROOT_DIRS[dataset]+filename,
            'caption': caption,
            'instruction': instruction,
            'text_inputs': text_inputs_input,
            'image_inputs': image_inputs_input,
            'seed': seed,
        })['description']

        print (predicted_label)

        predicted_label = clean_text(predicted_label).title()

        print ("Pred:", predicted_label, "Orig:", label)

        results.append([
            filename, caption, label, predicted_label
        ])
        write_csv(results_file, results)
   
def eval_model_cot_sl(
    model,
    dataset,
    instruction = INSTRUCTION_COT_SL_HARD['gpt4o'],
    text_inputs = text_inputs_clue_sl_hard,
    image_inputs = image_inputs_clue_sl_hard,
    shot = 'zero',
    metadata_path = "metadata.json",
    results_file = './outputs/results_cot.csv',
    seed = 42
):
    data = read_json(metadata_path)
    call_model = load_model(model)

    if(os.path.exists(results_file)):
        results = read_csv(results_file)
    else:
        results = [['Image', 'Caption', 'True Label', 'Predicted Label', 'Analysis']]

    total = len(data)
    for i, item in enumerate(data):
        print ()
        print(f"{i+1}/{total}")
        
        caption = item["caption"]
        image = item["filename"]

        label = item['labels'][0]

        text_inputs_input = [] if shot == 'zero' else text_inputs
        image_inputs_input = [] if shot == 'zero' else image_inputs

        if (i+1 < len(results)):
            continue

        analysis = call_model({
            'label_type': 'single_label',
            'text_inputs': text_inputs_input,
            'image_inputs': image_inputs_input,
            'image': DATASET_ROOT_DIRS[dataset]+image,
            'caption': caption,
            'instruction': instruction[0],
            'save_history': True,
            'seed': seed,
        })
        print (f"CoT step in {analysis['time']} sec:")
        print (analysis['description']+'\n')

        predicted_label = call_model({
            'label_type': 'single_label',
            'text_inputs': [],
            'image_inputs': [],
            'image': None,
            'caption': None,
            'instruction': instruction[1],
            'history': analysis['history'],
            'seed': seed,
        })['description']

        print (predicted_label)
        predicted_label = clean_text(predicted_label).title()

        print ("Pred:", predicted_label, "Orig:", label)
        results.append([
            image, caption, label, predicted_label, analysis['description']
        ])
        write_csv(results_file, results)

def eval_model_default_ml(
    model,
    dataset,
    text_inputs = (text_inputs_clue_ml, text_inputs_clue_ml_csv),
    image_inputs = image_inputs_clue_ml,
    instruction = INSTRUCTION_DEFAULT_ML['gpt4o'],
    shot = 'zero',
    metadata_path = "metadata.json",
    results_file = './outputs/results_default.csv',
    seed = 42
):
    data = read_json(metadata_path)
    call_model = load_model(model)

    if(os.path.exists(results_file)):
        results = read_csv(results_file)
    else:
        results = [['Image', 'Caption', 'True Labels', 'Predicted Labels']]

    total = len(data)
    for i, item in enumerate(data):
        print()
        print(f"{i+1}/{total}")
        caption = item["caption"]
        filename = item["filename"]

        labels = list(sorted(item["labels"]))

        text_inputs_input = []
        image_inputs_input = []

        if (shot == 'few'):
            image_inputs_input = image_inputs
            if (model not in CSV_OUTPUT_MODELS):
                # Chooses JSON based outputs for few shot
                text_inputs_input = text_inputs[0]
            else:
                # Chooses CSV based outputs for few shot
                text_inputs_input = text_inputs[1]

            # Remove one example, to avoid exceeding context length
            if (model == 'llava_ov'):
                text_inputs_input = text_inputs_input[1:]
                image_inputs_input = image_inputs_input[1:]

        if (i+1 < len(results)):
            continue

        predicted_labels_raw = call_model({
            'label_type': 'multi_label',
            'image': DATASET_ROOT_DIRS[dataset]+filename,
            'caption': caption,
            'instruction': instruction,
            'text_inputs': text_inputs_input,
            'image_inputs': image_inputs_input,
            'seed': seed,
        })['description']

        print (predicted_labels_raw)
        predicted_labels_clean = clean_json_string(predicted_labels_raw)

        try:
            predicted_labels = json.loads(predicted_labels_clean)["labels"]
        except:
            predicted_labels = []

        # Using this parsing, if model is prompted to give CSV output
        if (not predicted_labels and model in CSV_OUTPUT_MODELS):
            csv_list = predicted_labels_raw.split(',')
            predicted_labels = [clean_text(x.strip()) for x in csv_list]

        if (type(predicted_labels) == str):
            predicted_labels = [predicted_labels]

        predicted_labels.sort()
        predicted_labels = [label.title() for label in predicted_labels]

        print (f"Pred: {predicted_labels}, Orig: {labels}")
        results.append([
            filename, caption, labels, predicted_labels
        ])
        write_csv(results_file, results)

def eval_model_cot_ml(
    model,
    dataset,
    text_inputs = (text_inputs_clue_ml, text_inputs_clue_ml_csv),
    image_inputs = image_inputs_clue_ml,
    instruction = INSTRUCTION_COT_ML['gpt4o'],
    shot = 'zero',
    metadata_path = "metadata.json",
    results_file = './outputs/results_default.csv',
    seed = 42
):
    data = read_json(metadata_path)
    call_model = load_model(model)

    if(os.path.exists(results_file)):
        results = read_csv(results_file)
    else:
        results = [['Image', 'Caption', 'True Labels', 'Predicted Labels', 'Analysis']]

    total = len(data)
    for i, item in enumerate(data):
        print()
        print(f"{i+1}/{total}")
        caption = item["caption"]
        filename = item["filename"]

        labels = list(sorted(item["labels"]))

        text_inputs_input = []
        image_inputs_input = []

        if (shot == 'few'):
            if (model not in CSV_OUTPUT_MODELS):
                # Chooses JSON based outputs for few shot
                text_inputs_input = text_inputs[0]
                image_inputs_input = image_inputs[0]
            else:
                # Chooses CSV based outputs for few shot
                text_inputs_input = text_inputs[1]
                image_inputs_input = image_inputs[1]
            
            # Remove one example, to avoid exceeding context length
            if (model == 'llava_ov'):
                text_inputs_input = text_inputs_input[1:]
                image_inputs_input = image_inputs_input[1:]

        if (i+1 < len(results)):
            continue

        analysis = call_model({
            'label_type': 'multi_label',
            'text_inputs': text_inputs_input,
            'image_inputs': image_inputs_input,
            'image': DATASET_ROOT_DIRS[dataset]+filename,
            'caption': caption,
            'instruction': instruction[0],
            'save_history': True,
            'seed': seed,
        })
        print (f"CoT step in {analysis['time']} sec:")
        print (analysis['description']+'\n')

        predicted_labels_raw = call_model({
            'label_type': 'multi_label',
            'text_inputs': [],
            'image_inputs': [],
            'image': None,
            'caption': None,
            'instruction': instruction[1],
            'history': analysis['history'],
            'seed': seed,
        })['description']

        print (predicted_labels_raw)
        predicted_labels_clean = clean_json_string(predicted_labels_raw)

        try:
            predicted_labels = json.loads(predicted_labels_clean)["labels"]
        except:
            predicted_labels = ''
        
        # Using this parsing, if model is prompted to give CSV output
        if (not predicted_labels and model in CSV_OUTPUT_MODELS):
            csv_list = predicted_labels_raw.split(',')
            predicted_labels = [clean_text(x.strip()) for x in csv_list]
    
        if (type(predicted_labels) == str):
            predicted_labels = [predicted_labels]

        predicted_labels.sort()
        predicted_labels = [label.title() for label in predicted_labels]

        print (f"Pred: {predicted_labels}, Orig: {labels}")
        results.append([
            filename, caption, labels, predicted_labels, analysis['description']
        ])
        write_csv(results_file, results)

######################### Invocation #############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VLM models on CORDIAL Benchmark')
    parser.add_argument('--model', type=str, help='Model to evaluate', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', required=True, 
                        choices=['clue_ml', 'clue_sl_hard', 'disrel_sl', 'tweets_sl'])
    parser.add_argument('--shot', type=str, help='Shot type', required=True, choices=['zero', 'few'])
    parser.add_argument('--seed', type=int, help='Random seed', required=True)
    parser.add_argument('--cot', action='store_true', help='Run using Chain-of-Thought prompt')
    parser.add_argument('--results_file', metavar='text', nargs='?', help='Results file')
    args = parser.parse_args()

    results_path = f'./llm_outputs/{args.model}/{args.dataset}/'
    os.makedirs(results_path, exist_ok=True)

    if (args.dataset == 'clue_ml'):
        if (args.cot):
            if (args.results_file): 
                results_path = os.path.join(results_path, args.results_file)
            else: 
                results_path = os.path.join(results_path, f'results_{args.shot}_shot_CoT_{args.dataset}.csv')
            
            try:
                instruction = instruction_cot_mapping[args.dataset][args.model]
            except:
                instruction = instruction_cot_mapping[args.dataset]['gpt4o']
            
            eval_model_cot_ml(
                model=args.model,
                dataset=args.dataset,
                text_inputs=text_examples_mapping[args.dataset],
                image_inputs=image_examples_mapping[args.dataset],
                instruction=instruction,
                shot=args.shot,
                metadata_path=test_metadata_paths[args.dataset],
                results_file=results_path,
                seed=args.seed
            )
        else:
            if (args.results_file): 
                results_path = os.path.join(results_path, args.results_file)
            else: 
                results_path = os.path.join(results_path, f'results_{args.shot}_shot_{args.dataset}.csv')
            
            try:
                instruction = instruction_default_mapping[args.dataset][args.model]
            except:
                instruction = instruction_default_mapping[args.dataset]['gpt4o']

            eval_model_default_ml(
                model=args.model,
                dataset=args.dataset,
                text_inputs=text_examples_mapping[args.dataset],
                image_inputs=image_examples_mapping[args.dataset],
                instruction=instruction,
                shot=args.shot,
                metadata_path=test_metadata_paths[args.dataset],
                results_file=results_path,
                seed=args.seed
            )
    else:
        if (args.cot):
            if (args.results_file): 
                results_path = os.path.join(results_path, args.results_file)
            else: 
                results_path = os.path.join(results_path, f'results_{args.shot}_shot_CoT_{args.dataset}.csv')

            try:
                instruction = instruction_cot_mapping[args.dataset][args.model]
            except:
                instruction = instruction_cot_mapping[args.dataset]['gpt4o']
            
            eval_model_cot_sl(
                model=args.model,
                dataset=args.dataset,
                text_inputs=text_examples_mapping[args.dataset],
                image_inputs=image_examples_mapping[args.dataset],
                instruction=instruction,
                shot=args.shot,
                metadata_path=test_metadata_paths[args.dataset],
                results_file=results_path,
                seed=args.seed
            )
        else:
            if (args.results_file): 
                results_path = os.path.join(results_path, args.results_file)
            else: 
                results_path = os.path.join(results_path, f'results_{args.shot}_shot_{args.dataset}.csv')
            
            try:
                instruction = instruction_default_mapping[args.dataset][args.model]
            except:
                instruction = instruction_default_mapping[args.dataset]['gpt4o']

            eval_model_default_sl(
                model=args.model,
                dataset=args.dataset,
                text_inputs=text_examples_mapping[args.dataset],
                image_inputs=image_examples_mapping[args.dataset],
                instruction=instruction,
                shot=args.shot,
                metadata_path=test_metadata_paths[args.dataset],
                results_file=results_path,
                seed=args.seed
            )