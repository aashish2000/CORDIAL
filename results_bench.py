import argparse
import os
import torch
import logging
from helper import read_csv, write_csv
from configs import DATASET_LABELS
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from collections import defaultdict

RESULTS_FOLDER = './llm_outputs/'

files_to_ignore = {'disrel_sl': ['images/3067.jpg', 'images/12319.jpg', 'images/50846.jpg', 'images/12834.jpg', 'images/34066.jpg', 'images/35187.jpg', 'images/514.jpg', 'images/77920.jpg', 'images/65142.jpg', 'images/53761.jpg', 'images/62575.jpg', 'images/38989.jpg', 'images/69597.jpg', 'images/80125.jpg', 'images/74389.jpg', 'images/3636.jpg', 'images/87367.jpg', 'images/41681.jpg', 'images/41399.jpg', 'images/10972.jpg', 'images/80579.jpg', 'images/116044.jpg'], 'clue_ml': ['images/4000.jpg', 'images/4669.jpg', 'images/400.jpg', 'images/948.jpg', 'images/1336.jpg', 'images/53.jpg', 'images/3300.jpg', 'images/1545.jpg', 'images/4885.jpg', 'images/4677.jpg', 'images/827.jpg', 'images/4792.jpg', 'images/1844.jpg', 'images/4934.jpg', 'images/1224.jpg', 'images/4910.jpg', 'images/5023.jpg', 'images/2994.jpg', 'images/5266.jpg', 'images/4077.jpg', 'images/1293.jpg', 'images/712.jpg', 'images/5311.jpg', 'images/4591.jpg', 'images/2768.jpg', 'images/2452.jpg', 'images/2532.jpg', 'images/2322.jpg', 'images/906.jpg', 'images/3264.jpg', 'images/6.jpg', 'images/2695.jpg', 'images/5007.jpg', 'images/3482.jpg', 'images/471.jpg', 'images/2341.jpg', 'images/96.jpg', 'images/37.jpg', 'images/385.jpg', 'images/713.jpg', 'images/931.jpg', 'images/70.jpg', 'images/1024.jpg', 'images/4929.jpg', 'images/94.jpg', 'images/250.jpg', 'images/210.jpg', 'images/1056.jpg', 'images/4789.jpg', 'images/496.jpg', 'images/4473.jpg', 'images/2393.jpg', 'images/3178.jpg', 'images/5328.jpg', 'images/4583.jpg', 'images/1194.jpg', 'images/771.jpg', 'images/5278.jpg', 'images/1188.jpg', 'images/4258.jpg', 'images/2560.jpg', 'images/410.jpg', 'images/455.jpg', 'images/4980.jpg'], 'clue_sl_hard': ['images/4000.jpg', 'images/948.jpg', 'images/4704.jpg', 'images/4933.jpg', 'images/4598.jpg', 'images/1545.jpg', 'images/4677.jpg', 'images/4792.jpg', 'images/1224.jpg', 'images/4494.jpg', 'images/4825.jpg', 'images/4410.jpg', 'images/5177.jpg', 'images/140.jpg', 'images/4469.jpg', 'images/4939.jpg', 'images/6.jpg', 'images/4380.jpg', 'images/471.jpg', 'images/1097.jpg', 'images/4571.jpg', 'images/534.jpg', 'images/70.jpg', 'images/4681.jpg', 'images/2393.jpg', 'images/4588.jpg', 'images/1194.jpg', 'images/242.jpg', 'images/5278.jpg', 'images/2067.jpg', 'images/4111.jpg', 'images/4142.jpg'], 'tweets_sl': ['multimodal_discourse_dataset/xAZz1Otpkw.jpg', 'multimodal_discourse_dataset/yy3ClMACmN.jpg', 'multimodal_discourse_dataset/uyqgAmpaPZ.jpg', 'multimodal_discourse_dataset/rhsMxeK1Qg.jpg', 'multimodal_discourse_dataset/uhdFFx6EMa.jpg', 'multimodal_discourse_dataset/vDkN8MDKEz.jpg', 'multimodal_discourse_dataset/zbUF9URF0z.jpg', 'multimodal_discourse_dataset/zNzbbs7G2V.jpg', 'multimodal_discourse_dataset/x8KgTpvN95.jpg', 'multimodal_discourse_dataset/uxYDy7PwbI.jpg', 'multimodal_discourse_dataset/yDLdSsUf9t.jpg', 'multimodal_discourse_dataset/z6ra1Z5KIR.jpg', 'multimodal_discourse_dataset/vNZdErrmcV.jpg', 'multimodal_discourse_dataset/vAdkCaDrom.jpg', 'multimodal_discourse_dataset/rOjSEtSuY1.jpg', 'multimodal_discourse_dataset/vmg1Yw0YVh.jpg', 'multimodal_discourse_dataset/u2xRVbMYpZ.jpg', 'multimodal_discourse_dataset/uOoufJnaQA.jpg', 'multimodal_discourse_dataset/uYcObOixMu.jpg', 'multimodal_discourse_dataset/uqwd9mBu61.jpg', 'multimodal_discourse_dataset/u12WkMARQ6.jpg', 'multimodal_discourse_dataset/sPmmwVGGEV.jpg', 'multimodal_discourse_dataset/u5TbplpDoT.jpg', 'multimodal_discourse_dataset/wCyPWW2syU.jpg', 'multimodal_discourse_dataset/uwVSBNYG9E.jpg', 'multimodal_discourse_dataset/ruGwjosvG3.jpg', 'multimodal_discourse_dataset/quwIksnZBd.jpg']}

extra_files_to_ignore = {'disrel_sl': ['images/63836.jpg'], 'tweets_sl': ['multimodal_discourse_dataset/zPgIy9zmEH.jpg', 'multimodal_discourse_dataset/uksjFCJBUy.jpg'], 'clue_ml': ['images/1737.jpg', 'images/3571.jpg', 'images/5230.jpg', 'images/2199.jpg', 'images/4884.jpg', 'images/2506.jpg', 'images/4380.jpg', 'images/4369.jpg', 'images/5295.jpg', 'images/1548.jpg', 'images/4108.jpg', 'images/4420.jpg', 'images/4110.jpg', 'images/4123.jpg', 'images/3002.jpg', 'images/4602.jpg', 'images/5298.jpg', 'images/4774.jpg', 'images/4990.jpg', 'images/4588.jpg', 'images/4142.jpg', 'images/4569.jpg', 'images/4685.jpg', 'images/3309.jpg', 'images/2009.jpg', 'images/1246.jpg', 'images/846.jpg', 'images/2205.jpg', 'images/154.jpg', 'images/4979.jpg', 'images/4594.jpg', 'images/5204.jpg', 'images/4571.jpg', 'images/534.jpg', 'images/4652.jpg', 'images/4691.jpg', 'images/2055.jpg', 'images/3365.jpg', 'images/1508.jpg', 'images/513.jpg', 'images/181.jpg', 'images/5324.jpg', 'images/4141.jpg', 'images/5238.jpg', 'images/5012.jpg'], 'clue_sl_hard': ['images/2601.jpg', 'images/3264.jpg', 'images/1099.jpg', 'images/2468.jpg']}

overall_metrics = {'disrel_sl': [['Model', 'Similar', 'Complementary', 'Overall']], 'clue_ml': [['Model', 'Visible', 'Subjective', 'Action', 'Story', 'Meta', 'Overall']], 'clue_sl_hard': [['Model', 'Visible', 'Subjective', 'Action', 'Story', 'Meta', 'Overall']], 'tweets_sl': [['Model', 'Insertion', 'Concretization', 'Projection', 'Restatement', 'Extension', 'Overall']]}

model_list = {
        "llava16_7b": "LLaVA 1.6 7B",
        "llava16_13b": "LLaVA 1.6 13B",
        "llava16_results": "LLaVA 1.6 34B",
        "llava_ov": "LLaVA OneVision 7B",
        "qwen2": "Qwen2-VL 7B",
        "finetune": "FT-Llama 3.2 Vision 11B",
        "llama32": "Llama 3.2 Vision 11B",
        "phi35": "Phi3.5 Vision 4.2B",
        "internvl25": "InternVL 2.5 26B",
        "gpt4o": "GPT-4o",
        "gemini_flash": "Gemini 1.5 Flash",
        "gemini_results": "Gemini 1.5 Pro",
        "claude": "Claude 3.5 Sonnet v2",
        "clip_classifier": "CLIP Classifier",
    }

def merge_dicts(dict1, dict2):
    merged_dict = {}
    # Get the union of keys from both dictionaries
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        # Use a set to remove duplicates, then convert back to a list
        merged_dict[key] = list(set(dict1.get(key, []) + dict2.get(key, [])))

    return merged_dict

def format_floats(float_array):
    return [f"{round(float(num), 3):.3f}" for num in float_array]

def eval_single_label_perf(y_true, y_pred, num_classes, label_to_index_mapping):
    """
    Calculate overall and per-class metrics.

    Args:
        y_true (list or tensor): Ground truth labels (class indices).
        y_pred (list or tensor): Predicted labels (class indices or probabilities).
        num_classes (int): Number of classes in the classification task.
        label_to_index_mapping (dict): Mapping from class labels to class indices.

    Returns:
        A Pandas DataFrame containing overall and per-class Accuracy, Precision, Recall, and F1-Score.
    """
    # Convert to tensors if input is a list
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Ensure y_pred contains class indices (argmax if probabilities are provided)
    if y_pred.ndim > 1:  # Probabilities provided
        y_pred = torch.argmax(y_pred, dim=1)

    # Initialize metrics
    overall_accuracy = MulticlassAccuracy(num_classes=num_classes, average="weighted")
    overall_precision = MulticlassPrecision(num_classes=num_classes, average="weighted")
    overall_recall = MulticlassRecall(num_classes=num_classes, average="weighted")
    overall_weighted_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
    overall_macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    overall_micro_f1 = MulticlassF1Score(num_classes=num_classes, average="micro")

    per_class_accuracy = MulticlassAccuracy(num_classes=num_classes, average=None)
    per_class_precision = MulticlassPrecision(num_classes=num_classes, average=None)
    per_class_recall = MulticlassRecall(num_classes=num_classes, average=None)
    per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)

    # Compute metrics
    overall_metrics = {
        "Accuracy": overall_accuracy(y_pred, y_true).item(),
        "Precision": overall_precision(y_pred, y_true).item(),
        "Recall": overall_recall(y_pred, y_true).item(),
        "F1-Score": overall_weighted_f1(y_pred, y_true).item(),
        "Macro F1": overall_macro_f1(y_pred, y_true).item(),
        "Micro F1": overall_micro_f1(y_pred, y_true).item(),
    }

    per_class_metrics = {
        "Accuracy": per_class_accuracy(y_pred, y_true).tolist(),
        "Precision": per_class_precision(y_pred, y_true).tolist(),
        "Recall": per_class_recall(y_pred, y_true).tolist(),
        "F1-Score": per_class_f1(y_pred, y_true).tolist(),
    }

    results = [["Label", "Accuracy", "Precision", "Recall", "F1-Score", "Macro F1", "Micro F1"]]

    results.append([
        "Overall", 
        overall_metrics['Accuracy'], 
        overall_metrics['Precision'], 
        overall_metrics['Recall'],
        overall_metrics['F1-Score'],
        overall_metrics['Macro F1'],
        overall_metrics['Micro F1']
    ])
    for i in range(num_classes):
        results.append([list(label_to_index_mapping.keys())[i],
                        per_class_metrics['Accuracy'][i],
                        per_class_metrics['Precision'][i],
                        per_class_metrics['Recall'][i],
                        per_class_metrics['F1-Score'][i]])

    # Print metrics
    print("Overall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}", end=" ")

    print("\nPer-Class Metrics:")
    for i in range(num_classes):
        print(f"{list(label_to_index_mapping.keys())[i]}:")
        print(f"  Accuracy: {per_class_metrics['Accuracy'][i]:.4f}", end=" ")
        print(f"  Precision: {per_class_metrics['Precision'][i]:.4f}", end=" ")
        print(f"  Recall: {per_class_metrics['Recall'][i]:.4f}", end=" ")
        print(f"  F1-Score: {per_class_metrics['F1-Score'][i]:.4f}")

    return results

def eval_multi_label_perf(y_true, y_pred, num_labels, label_to_index_mapping):
    """
    Compute metrics for multi-label classification.

    Args:
        y_true: The true one-hot labels.
        y_pred: The predicted one-hot labels.
        num_labels: The number of labels in the multi-label classification task.
        label_to_index_mapping (dict): Mapping from class labels to class indices.

    Returns:
        A Pandas DataFrame containing overall and per-class Precision, Recall, and F1-score.
    """
    # Initialize torchmetrics metrics for overall and per-class
    overall_precision = MultilabelPrecision(num_labels=num_labels, average="weighted")
    overall_recall = MultilabelRecall(num_labels=num_labels, average="weighted")
    overall_f1 = MultilabelF1Score(num_labels=num_labels, average="weighted")
    overall_macro_f1 = MultilabelF1Score(num_labels=num_labels, average="macro")
    overall_micro_f1 = MultilabelF1Score(num_labels=num_labels, average="micro")

    per_class_precision = MultilabelPrecision(num_labels=num_labels, average=None)
    per_class_recall = MultilabelRecall(num_labels=num_labels, average=None)
    per_class_f1 = MultilabelF1Score(num_labels=num_labels, average=None)

    predictions = torch.tensor(y_pred).int()
    labels = torch.tensor(y_true).int()

    overall_precision.update(predictions, labels)
    overall_recall.update(predictions, labels)
    overall_f1.update(predictions, labels)
    overall_macro_f1.update(predictions, labels)
    overall_micro_f1.update(predictions, labels)

    per_class_precision.update(predictions, labels.int())
    per_class_recall.update(predictions, labels.int())
    per_class_f1.update(predictions, labels.int())

    # Compute overall metric values
    overall_metrics = {
        "Precision": overall_precision.compute().item(),
        "Recall": overall_recall.compute().item(),
        "F1-Score": overall_f1.compute().item(),
        "Macro F1": overall_macro_f1.compute().item(),
        "Micro F1": overall_micro_f1.compute().item(),
    }

    # Compute per-class metric values
    per_class_metrics = {
        "Precision": per_class_precision.compute().tolist(),
        "Recall": per_class_recall.compute().tolist(),
        "F1-Score": per_class_f1.compute().tolist(),
    }

    results = [["Label", "Precision", "Recall", "F1-Score", "Macro F1", "Micro F1"]]

    results.append([
        "Overall", 
        overall_metrics['Precision'], 
        overall_metrics['Recall'],
        overall_metrics['F1-Score'],
        overall_metrics['Macro F1'],
        overall_metrics['Micro F1']
    ])
    for i in range(num_labels):
        results.append([list(label_to_index_mapping.keys())[i],
                        per_class_metrics['Precision'][i],
                        per_class_metrics['Recall'][i],
                        per_class_metrics['F1-Score'][i]])

    # Display metrics
    print("Overall Evaluation Metrics:")
    print(f"Precision: {overall_metrics['Precision']:.4f}", end=" ")
    print(f"Recall: {overall_metrics['Recall']:.4f}", end=" ")
    print(f"F1-Score: {overall_metrics['F1-Score']:.4f}", end=" ")
    print(f"Macro F1: {overall_metrics['Macro F1']:.4f}", end=" ")
    print(f"Micro F1: {overall_metrics['Micro F1']:.4f}")

    print("\nPer-Class Evaluation Metrics:")
    for i in range(num_labels):
        print(f"{list(label_to_index_mapping.keys())[i]}:")
        print(f"  Precision: {per_class_metrics['Precision'][i]:.4f}", end=" ")
        print(f"  Recall: {per_class_metrics['Recall'][i]:.4f}", end=" ")
        print(f"  F1-Score: {per_class_metrics['F1-Score'][i]:.4f}")

    return results

def change_labels_to_onehot(lst, label_to_index_mapping):
    one_hot = [0] * len(label_to_index_mapping)
    for label in lst:
        try:
            one_hot[label_to_index_mapping[label]] = 1
        except KeyError:
            raise KeyError(f"Label {label} not found in label_to_index_mapping.")
    return one_hot

def change_label_to_index(label, label_to_index_mapping):
    return label_to_index_mapping[label]

def parse_and_eval(results_path, metrics_path, dataset):
    data = read_csv(results_path)[1:]
    test_set_results_true = []
    test_set_results_pred = []
    image_paths = []
    warning_paths = defaultdict(list)
    for item in data:
        image, _, true, pred = item[0:4]
        if(image not in files_to_ignore.get(dataset, []) and image not in extra_files_to_ignore.get(dataset, [])):
            test_set_results_true.append(true)
            test_set_results_pred.append(pred)
            image_paths.append(image)
    
    label_to_index_mapping = DATASET_LABELS[dataset]
    num_classes = len(label_to_index_mapping)

    num_warnings = 0
    if (dataset == 'clue_ml'):
        test_set_results_pred_mapped = []
        test_set_results_true_mapped = []
        for i, labels in enumerate(zip(test_set_results_true, test_set_results_pred)):
            true_labels, pred_labels = labels
            try:
                test_set_results_pred_mapped.append(change_labels_to_onehot(eval(pred_labels), label_to_index_mapping))
            except Exception:
                logging.warning(f"Predicted Label {pred_labels} (at {i+1}th row) not found in label_to_index_mapping. Skipping...")
                warning_paths[dataset].append(image_paths[i])
                num_warnings += 1
                continue

            test_set_results_true_mapped.append(change_labels_to_onehot(eval(true_labels), label_to_index_mapping))

        df = eval_multi_label_perf(test_set_results_true_mapped, test_set_results_pred_mapped, num_classes, label_to_index_mapping)
    else:
        test_set_results_true_mapped = []
        test_set_results_pred_mapped = []
        for i, labels in enumerate(zip(test_set_results_true, test_set_results_pred)):
            true_label, pred_label = labels
            try:
                test_set_results_pred_mapped.append(change_label_to_index(pred_label.title(), label_to_index_mapping))
            except KeyError:
                logging.warning(f"Predicted Label {pred_label} (at {i+1}th row) not found in label_to_index_mapping. Skipping...")
                warning_paths[dataset].append(image_paths[i])
                num_warnings += 1
                continue

            test_set_results_true_mapped.append(change_label_to_index(true_label, label_to_index_mapping))

        df = eval_single_label_perf(test_set_results_true_mapped, test_set_results_pred_mapped, num_classes, label_to_index_mapping)
    
    save_metrics = [None] * (len(overall_metrics[dataset][0]) - 2)
    for row in df[2:]:
        save_metrics[label_to_index_mapping[row[0]]] = row[-1]
    
    save_metrics = [results_path] + save_metrics + [df[1][-2]]
    overall_metrics[dataset].append(save_metrics)

    write_csv(metrics_path, df)
    print ("Number of warnings:", num_warnings)
    print ('Metrics saved to', metrics_path)
    return (warning_paths)

def traverse_and_eval_files(curr_results_path, dataset):
    all_warning_paths = defaultdict(list)
    for file in os.listdir(curr_results_path):
        if file.endswith('.csv') and file.startswith('results'):
            results_file = os.path.join(curr_results_path, file)
            metrics_file = os.path.join(curr_results_path, 'metrics_' + file)
            all_warning_paths = merge_dicts(all_warning_paths, parse_and_eval(results_file, metrics_file, dataset))
    
    return (all_warning_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics for VLM models.')
    parser.add_argument('--model', type=str, help='Model to evaluate')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', choices=['clue_ml', 'clue_sl_hard', 'disrel_sl', 'tweets_sl'])
    parser.add_argument('--results_file', metavar='text', nargs='?', help='Results file')
    parser.add_argument('--metrics_file', metavar='text', nargs='?', help='Name of metrics file')
    parser.add_argument('--all_files', action='store_true', help='Evaluate all results files for the model and dataset')
    parser.add_argument('--all_datasets', action='store_true', help='Evaluate all datasets for the model')
    parser.add_argument('--all_models', action='store_true', help='Evaluate all datasets and models in results folder')

    args = parser.parse_args()

    all_warning_paths_all_datasets = defaultdict(list)
    
    if (args.all_models):
        for model in os.listdir(RESULTS_FOLDER):
            results_path = os.path.join(RESULTS_FOLDER, model)
            for dataset in DATASET_LABELS.keys():
                curr_results_path = os.path.join(results_path, dataset)
                if (not os.path.exists(curr_results_path)):
                    logging.warning( f"Results path {curr_results_path} does not exist. Skipping dataset...")
                    continue
                all_warning_paths_all_datasets = merge_dicts(all_warning_paths_all_datasets, traverse_and_eval_files(curr_results_path, dataset))
    
    else:
        results_path = os.path.join(RESULTS_FOLDER, args.model)
        if (not os.path.exists(results_path)):
            raise FileNotFoundError(f"Results path {results_path} does not exist.")
        
        if (args.all_datasets):
            for dataset in DATASET_LABELS.keys():
                curr_results_path = os.path.join(results_path, dataset)
                if (not os.path.exists(curr_results_path)):
                    logging.warning( f"Results path {curr_results_path} does not exist. Skipping dataset...")
                    continue
                all_warning_paths_all_datasets = merge_dicts(all_warning_paths_all_datasets, traverse_and_eval_files(curr_results_path, dataset))
        
        elif (args.all_files):
            if (args.dataset is None):
                raise RuntimeError("Dataset must be specified when using --all_files.")

            curr_results_path = os.path.join(results_path, args.dataset)
            all_warning_paths_all_datasets = merge_dicts(all_warning_paths_all_datasets, traverse_and_eval_files(curr_results_path, args.dataset))

        else:
            if (args.dataset is None):
                raise RuntimeError("Dataset must be specified if --all_files or --all_datasets is not used.")
            
            if (args.results_file is None):
                raise RuntimeError("Results file must be specified if --all_files or --all_datasets is not used.")

            results_file = os.path.join(results_path, args.dataset, args.results_file)
            if (args.metrics_file):
                metrics_file = os.path.join(results_path, args.dataset, args.metrics_file)
            else:
                metrics_file = os.path.join(results_path, args.dataset, 'metrics_' + args.results_file)

            all_warning_paths_all_datasets = merge_dicts(all_warning_paths_all_datasets, parse_and_eval(results_file, metrics_file, args.dataset))

    print ("All warning paths:", all_warning_paths_all_datasets)
    print([(key, len(value)) for key, value in all_warning_paths_all_datasets.items()])