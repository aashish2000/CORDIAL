import os, sys
import google.auth
import google.auth.transport.requests
import openai
import argparse
from helper import retry_if_fail
from anthropic import AnthropicVertex
from prompt_builder import process_text, process_image_openai, process_image_anthropic
from helper import read_json, read_csv, write_csv
from configs import test_metadata_paths
from environment import DATASET_ROOT_DIRS
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from environment import LOCATION, OPENAI_API_KEY

ANALYSIS_PROMPT = {
    'clue_ml' : [
        {
            "role" : "system",
            "content" : [
                process_text((
                    "You are an expert linguist. Your task is to analyze a image-text pair and verify if the given Coherence Relations are appropriate.\n"
                    "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
                    "These are the possible coherence relations you can assign to an image-text pair:\n"
                    "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
                    "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
                    "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
                    "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
                    "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
                ))
            ] 
        },
        {
            "role" : "user",
            "content" : [
                process_text("Based on provided information, reply True (if appropriate) or False (if not appropriate) for the following image-text pair. Give your rationale behind it.\n"),
            ]
        }
    ],
    'clue_sl_hard' : [
        {
            "role" : "system",
            "content" : [
                process_text((
                    "You are an expert linguist. Your task is to analyze a image-text pair and verify if the given Coherence Relation is appropriate.\n"
                    "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
                    "These are the possible coherence relations you can assign to an image-text pair:\n"
                    "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
                    "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
                    "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
                    "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
                    "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
                ))
            ] 
        },
        {
            
            "role" : "user",
            "content" : [
                process_text("Based on provided information, reply True (if appropriate) or False (if not appropriate) for the following image-text pair. Give your rationale behind it.\n"),
            ]
        }
    ],
    'tweets_sl' : [
        {
            "role" : "system",
            "content" : [
                process_text((
                    "You are an expert linguist. Your task is to analyze a image-text pair and verify if the given Coherence Relation is appropriate.\n"
                    "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
                    "These are the possible coherence relations you can assign to an image-text pair:\n"
                    "- Insertion: The salient object described in the image is not explicitly mentioned in the text.\n"
                    "- Concretization: Both the text and image contain a mention of the main visual entity.\n"
                    "- Projection: The main entity mentioned in the text is implicitly related to the visual objects present in the image.\n"
                    "- Restatement: The text directly describes the image contents.\n"
                    "- Extension: The image expands upon the story or idea in the text, presenting new elements or elaborations, effectively filling in narrative gaps left by the text.\n"
                ))
            ] 
        },
        {
            "role" : "user",
            "content" : [
                process_text("Based on provided information, reply True (if appropriate) or False (if not appropriate) for the following image-text pair. Give your rationale behind it.\n"),
            ]
        }
    ],
    'disrel_sl' : [
        {
            "role" : "system",
            "content" : [
                process_text((
                    "You are an expert linguist. Your task is to analyze a image-text pair and verify if the given Coherence Relation is appropriate.\n"
                    "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
                    "These are the possible coherence relations you can assign to an image-text pair:\n"
                    "- Similar: The image and text provide the same information and share the same focus. There exists significant overlap in information conveyed between modalities.\n"
                    "- Complementary: The image and text do not provide the same information or share the same focus but one modality helps understand the other better.\n"
                ))
            ] 
        },
        {
            
            "role" : "user",
            "content" : [
                process_text("Based on provided information, reply True (if appropriate) or False (if not appropriate) for the following image-text pair. Give your rationale behind it.\n"),
            ]
        }
    ],
}

def refresh_creds(creds):
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

# Setup client with Google Cloud credentials
def load_gemini():
    creds, project_id = google.auth.default()
    refresh_creds(creds)

    client = openai.OpenAI(
        base_url=f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{LOCATION}/endpoints/openapi",
        api_key=creds.token
    )
    return client, creds

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
def call_gemini(client, creds, prompt, seed):
    if (creds.expired):
        refresh_creds(creds)
        client.api_key = creds.token

    response = client.chat.completions.create(
        model = 'google/gemini-1.5-pro-002',
        messages = prompt,
        seed = seed
    )
    answer = (response.choices[0].message.content)
    return answer

@retry_if_fail
def call_gpt4o(client, prompt, seed):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = prompt,
        seed = seed
    ) 
    answer = (response.choices[0].message.content)
    return answer

@retry_if_fail
def call_claude(client, creds, system_msg, prompt):
    if (creds.expired):
        refresh_creds(creds)
        client.api_key = creds.token

    response = client.messages.create(
        model = 'claude-3-5-sonnet-v2@20241022',
        messages = prompt,
        system = system_msg,
        max_tokens = 512,
    )
    answer = response.content[0].text
    return answer

def calculate_results(results_file):
    results = read_csv(results_file)
    total = len(results) - 1
    incorrect = 0
    for row in results[1:]:
        if (row[3].lower().startswith('false')):
            incorrect += 1
        elif ('false' in row[3].lower()):
            incorrect += 1

    print (f"Incorrect: {incorrect} ({incorrect / total * 100:.2f}%) out of {total}")

def calculate_results_in_folder(results_folder):
    for file in sorted(os.listdir(results_folder)):
        if file.endswith('.csv'):
            print (file, end=" - ")
            calculate_results(os.path.join(results_folder, file))

def eval_model_default_sl(
    model,
    root_dir,
    instruction,
    metadata_path = "metadata.json",
    results_file = './outputs/results_default.csv',
    seed = 42
):
    data = read_json(metadata_path)
    if model == 'gemini':
        client, creds = load_gemini()
    elif model == 'gpt4o':
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    elif model == 'claude':
        client, creds = load_claude()

    if(os.path.exists(results_file)):
        results = read_csv(results_file)
    else:
        results = [['Image', 'Caption', 'True Labels', 'Yes/No']]

    total = len(data)
    for i, item in enumerate(data):
        if (i+1 < len(results)):
            continue

        print()
        print(f"{i+1}/{total}")

        caption = item["caption"]
        filename = item["filename"]

        # Use CLUE resized image if original image is too large
        # Base64 encoding has some overhead, so we keep 4Mb instead of 5Mb
        if (os.stat(root_dir + filename).st_size > 4000000):
            root_dir = root_dir.replace('raw_dataset', 'resized_dataset')

        label = item['labels'][0]

        prompt = instruction[:]

        if (model == 'gemini'):
            prompt.append({
                "role" : "user",
                "content" : [
                    process_image_openai(root_dir + filename, image_mode = 'path', image_input_detail = 'low'),
                    process_text(f"Caption: {caption}"),
                    process_text(f"Coherence Relation: {label}"),
                ]
            })
            answer = call_gemini(client, creds, prompt, seed)

        elif (model == 'gpt4o'):
            prompt.append({
                "role" : "user",
                "content" : [
                    process_image_openai(root_dir + filename, image_mode = 'path', image_input_detail = 'low'),
                    process_text(f"Caption: {caption}"),
                    process_text(f"Coherence Relation: {label}"),
                ]
            })
            answer = call_gpt4o(client, prompt, seed)

        elif (model == 'claude'):
            prompt.append({
                "role" : "user",
                "content" : [
                    process_image_anthropic(root_dir + filename, image_mode = 'path', image_input_detail = 'low'),
                    process_text(f"Caption: {caption}"),
                    process_text(f"Coherence Relation: {label}"),
                ]
            })
            system_msg = prompt[0]['content'][0]['text']
            prompt = prompt[1:]

            answer = call_claude(client, creds, system_msg, prompt)

        print (answer)
        results.append([
            filename, caption, label, answer
        ])
        write_csv(results_file, results)

def eval_model_default_ml(
    model,
    root_dir,
    instruction,
    metadata_path = "metadata.json",
    results_file = './outputs/results_default.csv',
    seed = 42
):
    data = read_json(metadata_path)
    if model == 'gemini':
        client, creds = load_gemini()
    elif model == 'gpt4o':
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    elif model == 'claude':
        client, creds = load_claude()

    if(os.path.exists(results_file)):
        results = read_csv(results_file)
    else:
        results = [['Image', 'Caption', 'True Labels', 'Yes/No']]

    total = len(data)
    for i, item in enumerate(data):
        if (i+1 < len(results)):
            continue

        print()
        print(f"{i+1}/{total}")
        
        caption = item["caption"]
        filename = item["filename"]

        # Use CLUE resized image if original image is too large
        # Base64 encoding has some overhead, so we keep 4Mb instead of 5Mb
        if (os.stat(root_dir + filename).st_size > 4000000):
            root_dir = root_dir.replace('raw_dataset', 'resized_dataset')

        labels = list(sorted(item["labels"]))
        labels_csv = ', '.join(labels)

        prompt = instruction[:]

        if (model == 'gemini'):
            prompt.append({
                "role" : "user",
                "content" : [
                    process_image_openai(root_dir + filename, image_mode = 'path', image_input_detail = 'low'),
                    process_text(f"Caption: {caption}"),
                    process_text(f"Coherence Relations: {labels_csv}"),
                ]
            })
            answer = call_gemini(client, creds, prompt, seed)

        elif (model == 'gpt4o'):
            prompt.append({
                "role" : "user",
                "content" : [
                    process_image_openai(root_dir + filename, image_mode = 'path', image_input_detail = 'low'),
                    process_text(f"Caption: {caption}"),
                    process_text(f"Coherence Relations: {labels_csv}"),
                ]
            })
            answer = call_gpt4o(client, prompt, seed)

        elif (model == 'claude'):
            prompt.append({
                "role" : "user",
                "content" : [
                    process_image_anthropic(root_dir + filename, image_mode = 'path', image_input_detail = 'low'),
                    process_text(f"Caption: {caption}"),
                    process_text(f"Coherence Relations: {labels_csv}"),
                ]
            })
            if (creds.expired):
                refresh_creds(creds)
                client.api_key = creds.token

            system_msg = prompt[0]['content'][0]['text']
            prompt = prompt[1:]

            answer = call_claude(client, creds, system_msg, prompt)

        print (answer)
        results.append([
            filename, caption, labels, answer
        ])
        write_csv(results_file, results)


######################### Invocation #############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VLM models on CORDIAL Benchmark for verification of Coherence Relations')
    parser.add_argument(
        "--model", choices=["gpt4o", "gemini", "claude"], required=True,
        help="Model to use for evaluation"
    )
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', required=True, 
                        choices=['clue_ml', 'clue_sl_hard', 'disrel_sl', 'tweets_sl'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--results_file', type=str, default='./outputs/results_default.csv', help='File to save results')
    parser.add_argument('--calculate_results', action='store_true', help='Calculate results from existing CSV files')
    args = parser.parse_args()

    results_dir = os.path.join('./llm_outputs_verification', args.model)
    os.makedirs(results_dir, exist_ok=True)

    if (args.calculate_results):
        calculate_results_in_folder(results_dir)
        sys.exit(0)

    if args.results_file:
        results_path = os.path.join(results_dir, args.results_file)
    else:
        results_path = os.path.join(results_dir, f'results_{args.dataset}.csv')

    instruction = ANALYSIS_PROMPT[args.dataset]
    if args.dataset == 'clue_ml':
        eval_model_default_ml(
            model=args.model,
            root_dir=DATASET_ROOT_DIRS['clue_ml'],
            instruction=instruction,
            metadata_path=test_metadata_paths['clue_ml'],
            results_file=results_path,
            seed=args.seed
        )
    else:
        eval_model_default_sl(
            model=args.model,
            root_dir=DATASET_ROOT_DIRS[args.dataset],
            instruction=instruction,
            metadata_path=test_metadata_paths[args.dataset],
            results_file=results_path,
            seed=args.seed
        )