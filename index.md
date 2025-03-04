---
layout: project_page
permalink: /

title: "CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?"

authors: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee

affiliations: The Pennsylvania State University, National Institute of Technology Tiruchirappalli

paper: https://www.arxiv.org/pdf/2502.11300
# video: https://www.youtube.com/results?search_query=
code: https://github.com/aashish2000/CORDIAL
# data: https://huggingface.co/docs/datasets
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
        Multimodal Large Language Models (MLLMs) are renowned for their superior instruction following and reasoning capabilities across diverse problem domains. However, existing benchmarks primarily focus on assessing factual and logical correctness in downstream tasks, with limited emphasis on evaluating MLLMs’ ability to interpret pragmatic cues and intermodal relationships. To address this gap, we assess the competency of MLLMs in performing Multimodal Discourse Analysis (MDA) using Coherence Relations. Our benchmark, CORDIAL, encompasses a broad spectrum of Coherence Relations across 3 different discourse domains at varying levels of granularity. Through our experiments on 10+ MLLMs employing different prompting strategies, we show that even top models like Gemini 1.5 Pro and GPT-4o fail to match the performance of simple classifier-based baselines. This study emphasizes the need to move beyond similarity-based metrics and adopt a discourse-driven framework for evaluating MLLMs, providing a more nuanced assessment of their capabilities.
        </div>
    </div>
</div>

---

![CORDIAL Main Architecture](/static/image/CORDIAL-Main.jpg)

Figure 1: CORDIAL presents a combination of literal & pragmatic relations for analyzing the intermodal reasoning capabilities of MLLMs. We evaluate MLLMs on the task of Multimodal Discourse Analysis through the prediction and verification of Coherence Relations.

## Contributions
1. We propose CORDIAL (**CO**herence **R**elations in **D**iscourse for **I**mages **A**nd **L**anguage), the first benchmark for evaluating MLLMs for Multi-modal Discourse Analysis (MDA) using Coherence Relations.
2. Our experiments show that MLLMs struggle to predict and verify Coherence Relations, especially when these relations are more pragmatic. 
3. We demonstrate the need for coherence-aware fine-tuning approaches to improve intermodal reasoning approaches of MLLMs.

## Coherence Relations
Coherence Relations define how different parts of a discourse are logically connected. They help in structuring information flow, allowing multimodal models to process and interpret complex relationships between text and images effectively. In CORDIAL, coherence relations can be categorized into:
- **Literal Relations**: These involve a direct alignment between the image and text, where both modalities explicitly convey the same message. (e.g., Restatement, Concretization)
- **Pragmatic Relations**: These require deeper inference, where meaning is derived from contextual cues rather than direct representation. (e.g., Projection, Extension, Meta).

Check out the image below for different examples of Coherence Relations in CORDIAL. For a more detailed overview of this concept, please check out our paper!

---

![Types of Coherence Relations](/static/image/coh-relations.png)
Table 1: Examples from each dataset in CORDIAL for all Coherence Relations. The words in <span style="color:red">red</span> are important cues present in the caption, while the words in <span style="color:orange">orange</span> show pragmatic cues inferred from the image-text pair.

## The CORDIAL Benchmark
Our benchmark consists of different datasets with human-annotated coherence relations, across three discourse domains. 

1. **DisREL** (Disaster Management): Explored the relationship of image-text pairs in disaster-related tweets.

2. **Tweet Subtitles** (Social Media): Measured cross-modal coherence relations on tweets related to open-domain topics.

3. **CLUE** (Online Articles): Extended text-only coherence relations to the multimodal setting. We create 2 settings: CLUE *Single-Label and Multi-Label* for our evaluation.

We evaluate **12 MLLMs** on the prediction and verification tasks on this benchmark. Different prompting strategies (zero-shot, few-shot and CoT) were also used to evaluate MLLMs. We also create a simple generalizable classifier, which utilizes CLIP in a zero-shot manner along with an MLP, for the prediction task.

## Results
We report per-class and overall **Macro F1** scores across all 4 settings for the evaluation task. **Response accuracy** was used as a measure of performance for the prediction task.

1. *MLLMs struggle with predicting Coherence Relations:* No MLLM shows improvements over the baseline classifier across all settings. Claude 3.5 Sonnet v2 and Gemini 1.5 Flash were among the best performing models.

2. *Pragmatic Relations are challenging:* MLLMs aren't able to accurately predict relations such as Insertion and Projection.

3. *Verification Accuracy depends on settings:* Performance of MLLMs are significantly lower when verifying non-literal relations, when compared to more literal ones.

4. *Inconsistency of Prompting Strategies:* Even with additional training examples or reasoning steps, Coherence Relation Prediction is a fundamentally difficult task that cannot be taught to MLLMs only through prompting strategies.

5. *Fine-tuning improves MLLM Reasoning:* MLLMs are able to learn to recognize coherence relations better when fine-tuned on a task-specific dataset.

6. *Model biases inhibit prediction performance:* MLLMs are biased towards certain relation categories, despite providing few-shot examples and prompt optimization strategies.

## Citation
```
@misc{ramakrishnan2025cordial,
      title={CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?}, 
      author={Aashish Anantha Ramakrishnan and Aadarsh Anantha Ramakrishnan and Dongwon Lee},
      year={2025},
      eprint={2502.11300},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11300}, 
}
```
