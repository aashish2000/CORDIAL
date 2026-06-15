# CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?

CORDIAL is a benchmark for multimodal discourse analysis using coherence relations.

- [Website](https://aashish2000.github.io/CORDIAL)
- [Paper](https://aclanthology.org/2025.acl-long.1033/)
- [Dataset](https://huggingface.co/datasets/aashananth/CORDIAL)

## Setup and Installation

### 1) Clone the repository
```bash
git clone https://github.com/aashish2000/CORDIAL.git
cd CORDIAL
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

For vLLM-based models (`phi35`, `llava_ov`, `internvl25`):
```bash
pip install -r requirements_vllm.txt
```

### 3) Download the CORDIAL benchmark dataset
```bash
pip install huggingface-hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='aashananth/CORDIAL',
    repo_type='dataset',
    local_dir='./data/CORDIAL',
    local_dir_use_symlinks=False
)
"
```

Expected folder layout:
- `./data/CORDIAL/clue_ml/`
- `./data/CORDIAL/clue_sl_hard/`
- `./data/CORDIAL/disrel_sl/`
- `./data/CORDIAL/tweets_sl/`

### 4) Configure `environment.py`
```bash
cp environment.py.example environment.py
```

Then edit `environment.py`:
```python
DATASET_ROOT_DIRS = {
    'clue_ml'      : './data/CORDIAL/clue_ml/',
    'clue_sl_hard' : './data/CORDIAL/clue_sl_hard/',
    'disrel_sl'    : './data/CORDIAL/disrel_sl/',
    'tweets_sl'    : './data/CORDIAL/tweets_sl/',
}

OPENAI_API_KEY = 'your_openai_api_key_here'   # needed for gpt4o
GEMINI_API_KEY = 'your_gemini_api_key_here'   # optional / not used by current Gemini loader
LOCATION = 'us-central1'                       # required for gemini/gemini_flash/claude loaders
```

> `gemini`, `gemini_flash`, and `claude` use Vertex AI credentials (`google.auth.default()`), so you also need ADC configured (for example via `gcloud auth application-default login`).

## Usage

### Evaluate models
Run:
```bash
python eval_vlm_clue.py --model <model_name> --dataset <dataset_name> --shot <zero|few> --seed <int> [--cot] [--results_file <filename.csv>]
```

Supported `--model` values:
- `llava16`
- `llava16_13b`
- `llava16_7b`
- `llava_ov`
- `qwen2`
- `llama32`
- `phi35`
- `internvl25`
- `gpt4o`
- `gemini`
- `gemini_flash`
- `claude`

Supported `--dataset` values:
- `clue_ml`
- `clue_sl_hard`
- `disrel_sl`
- `tweets_sl`

Examples:
```bash
# Zero-shot on CLUE multi-label
python eval_vlm_clue.py --model llava16_7b --dataset clue_ml --shot zero --seed 42

# Few-shot + CoT
python eval_vlm_clue.py --model qwen2 --dataset clue_sl_hard --shot few --seed 42 --cot

# GPT-4o on DisRel
python eval_vlm_clue.py --model gpt4o --dataset disrel_sl --shot zero --seed 42
```

### Analyze evaluation results
`results_bench.py` does **not** take `--shot` or `--cot`. It evaluates result CSV files from `./llm_outputs/`.

Single file:
```bash
python results_bench.py --model qwen2 --dataset clue_sl_hard --results_file results_few_shot_CoT_clue_sl_hard.csv
```

All files for one model+dataset:
```bash
python results_bench.py --model qwen2 --dataset clue_sl_hard --all_files
```

All datasets for one model:
```bash
python results_bench.py --model qwen2 --all_datasets
```

All models and datasets:
```bash
python results_bench.py --all_models
```

## Fine-tuning (Llama 3.2 Vision)

Scripts available:
- `llama32_finetuning.py`
- `llama32_finetune_inference.py`

Current status: these scripts parse `--dataset` but internally reference `args.dataset_name`, which causes runtime failure unless corrected in code. If you want, I can patch these scripts next.

## Output Structure

Evaluation outputs are saved under:
```text
llm_outputs/
└── <model_name>/
    └── <dataset_name>/
        ├── results_zero_shot_<dataset_name>.csv
        ├── results_few_shot_<dataset_name>.csv
        ├── results_zero_shot_CoT_<dataset_name>.csv
        └── results_few_shot_CoT_<dataset_name>.csv
```

## News 🚀
- [2025-07-27] The benchmark dataset and code has been published.
- [2025-05-17] CORDIAL has been accepted to [ACL (Main) 2025](https://2025.aclweb.org/).
- [2025-02-16] Our paper is available on [arXiv](https://arxiv.org/abs/2502.11300).

## Citing
If you find our work useful, please cite:
```bibtex
@inproceedings{anantha-ramakrishnan-etal-2025-cordial,
    title = "{CORDIAL}: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?",
    author = "Anantha Ramakrishnan, Aashish  and
      Ramakrishnan, Aadarsh Anantha  and
      Lee, Dongwon",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1033/",
    doi = "10.18653/v1/2025.acl-long.1033",
    pages = "21277--21297",
    ISBN = "979-8-89176-251-0"
}
```
