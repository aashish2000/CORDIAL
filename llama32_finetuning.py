import argparse
from helper import read_json
from prompt_builder import build_sl_prompt, build_ml_prompt
from configs import train_metadata_paths, instruction_default_mapping
from environment import DATASET_ROOT_DIRS
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

def load_custom_dataset(metadata_path, root_dir, instruction, dataset_name):
    """Loads the dataset based on metadata and prepares messages for the model."""
    metadata = read_json(metadata_path)
    all_messages = []
    for item in metadata:
        caption = item["caption"]
        filename = item["filename"]
        image = root_dir + filename

        if (dataset_name == 'clue_ml'):
            label = ', '.join(item['labels'])
            messages = build_ml_prompt(
                text_inputs=[], image_inputs=[], caption=caption, image=image,
                image_mode='path', image_input_detail='high', instruction=instruction,
                unsloth_finetuning=True,
            )
            messages[-1]['content'] += label
        else:
            label = item['labels'][0]
            messages = build_sl_prompt(
                text_inputs=[], image_inputs=[], caption=caption, image=image,
                image_mode='path', image_input_detail='high', instruction=instruction,
                unsloth_finetuning=True,
            )
            messages[-1]['content'][0]['text'] += label

        # System messages not allowed
        messages[0]['role'] = 'user'
        all_messages.append({'messages': messages})
    return all_messages

def main(args):
    """Main function to run the training process."""
    # --- 1. Load Dataset ---
    try:
        instruction = instruction_default_mapping[args.dataset_name]['ft_llama32']
    except KeyError:
        instruction = instruction_default_mapping[args.dataset_name]['gpt4o']

    train_dataset = load_custom_dataset(
        train_metadata_paths[args.dataset_name],
        DATASET_ROOT_DIRS[args.dataset_name],
        instruction,
        args.dataset_name
    )

    # --- 2. Load Model & Tokenizer ---
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        use_gradient_checkpointing="unsloth",
        load_in_4bit=args.load_in_4bit,
    )

    # --- 3. Configure PEFT Model (LoRA) ---
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None
    )

    # --- 4. Set up Trainer ---
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=10,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
            save_strategy="epoch",
            # Required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=args.max_seq_length,
        ),
    )

    # --- 5. Start Training ---
    trainer.train()

    # --- 6. Save Final Model ---
    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)
    print(f"✅ Model and tokenizer successfully saved to {args.save_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a Vision Language Model with configurable hyperparameters.")

    # --- Path and Naming Arguments ---
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate', required=True, 
                        choices=['clue_ml', 'clue_sl_hard', 'disrel_sl', 'tweets_sl'])    
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.2-11B-Vision-Instruct', help='Base model name or path from Hugging Face.')
    parser.add_argument('--output_dir', type=str, default='outputs/checkpoints', help='Directory to save training outputs and checkpoints.')
    parser.add_argument('--save_model_dir', type=str, default='outputs/final_model', help='Directory to save the final finetuned model.')

    # --- Model Loading and PEFT Arguments ---
    parser.add_argument('--load_in_4bit', action='store_true', help='Load the base model in 4-bit precision.')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (r).')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout.')
    parser.add_argument('--bias', type=str, default='none', help='Bias type for LoRA ("none", "all", or "lora_only").')
    parser.add_argument('--no-use_rslora', dest='use_rslora', action='store_false', help='Disable the use of Rank-Stabilized LoRA.')
    parser.set_defaults(use_rslora=True)

    # --- Layer Finetuning Arguments (use --no-<option> to disable) ---
    parser.add_argument('--no-finetune_vision_layers', dest='finetune_vision_layers', action='store_false', help='Do not finetune vision layers.')
    parser.add_argument('--no-finetune_language_layers', dest='finetune_language_layers', action='store_false', help='Do not finetune language layers.')
    parser.add_argument('--no-finetune_attention_modules', dest='finetune_attention_modules', action='store_false', help='Do not finetune attention modules.')
    parser.add_argument('--no-finetune_mlp_modules', dest='finetune_mlp_modules', action='store_false', help='Do not finetune MLP modules.')
    parser.set_defaults(finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True)

    # --- Training Hyperparameter Arguments ---
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, help='Batch size per device during training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Total number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Initial learning rate for the optimizer.')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length for the model.')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps for the learning rate scheduler.')
    parser.add_argument('--optim', type=str, default='adamw_torch', help='Optimizer to use (e.g., "adamw_torch").')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type (e.g., "linear", "cosine").')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args)