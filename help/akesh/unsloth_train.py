import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mlflow
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from SinhalaSpellChecker.utils.google import upload_folder
import torch
from unsloth import  is_bfloat16_supported
from SinhalaSpellChecker.Models.SpellCheckTrainer import SpellCheckTrainer
import pandas as pd
from SinhalaSpellChecker.Unsloth.unsloth import get_model, prepare_datasets_with_formatting, prepare_datasets, test_model, log_and_save_results

def main(args):
    
    args.max_seq_length = 2048
    if torch.cuda.get_device_name(0).startswith('Tesla T4'):
        args.dtype = torch.float16
    else:
        args.dtype = torch.bfloat16
    # dtype = None# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    args.load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    exp_name = args.exp_name
    saving_steps = args.save_steps
    checkpoint_path = None
    model, tokenizer = get_model(args.model_name, args.max_seq_length, args.dtype, args.load_in_4bit, args.seed)
    train_dataset, val_dataset, test = prepare_datasets(tokenizer.eos_token)
    # train_dataset, val_dataset, test = prepare_datasets_with_formatting(tokenizer.eos_token)
    
    def collate_fn_complex(examples):
        alpaca_prompt = """You are an Expert Sinhala Spell Corrector. Below is a sentence in Sinhala Language. It may or may not have a spelling mistake. Give the corrected output in Sinhala.

        ### Text:
        {}
        
        ### Output:
        {}"""
        EOS_TOKEN = tokenizer.eos_token
        texts = [alpaca_prompt.format(example['text'], example['expected']) + EOS_TOKEN for example in examples]
        
        batch = tokenizer(texts, return_tensors="pt", padding=True)
        if batch["input_ids"].size(1) > args.max_seq_length:
            batch["input_ids"] = batch["input_ids"][:, :args.max_seq_length]
            batch["attention_mask"] = batch["attention_mask"][:, :args.max_seq_length]
        labels = batch["input_ids"].clone()

        '''
        Loss function improvement.
        '''
        for i, example in enumerate(examples):
            
            prompt = alpaca_prompt.format(example['text'], "")
            prompt_len = len(tokenizer(prompt)['input_ids'])
            
            start_idx = prompt_len            
            
            # Mask everything outside the expected range to -100
            labels[i, :start_idx] = -100

        # Update the labels in the batch
        
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
            
        return batch
    
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        # packing=True,
        report_to='none',
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1000,  # Logs every 1000 steps
        eval_steps=saving_steps,  # Evaluate every 10,000 steps
        eval_strategy="steps",  # Perform evaluation based on steps
        save_steps=saving_steps,  # Save the model every 10,000 steps
        save_total_limit=4,  # Keep only the last 3 checkpoints
        load_best_model_at_end=True,  # Load the best model after training ends
        metric_for_best_model="eval_loss",  # Use eval_loss for early stopping
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        output_dir=exp_name,
        seed=args.seed,
        remove_unused_columns = False,
        dataset_kwargs = {"skip_prepare_dataset": True},
    )
    
    trainer = SpellCheckTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator = collate_fn_complex,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Set the validation set for evaluation
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=os.cpu_count(),
        args=training_args,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping with patience
    )

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    from SinhalaSpellChecker.util_classes.logger import Logger
    import time
    logger = Logger(exp_name, None)
    logger.log_unsloth_params(args)
    
    trainer_stats = trainer.train()

    logger.log_unsloth_metrics(trainer_stats, trainer)
    torch.cuda.empty_cache()

    model.save_pretrained(exp_name)
    tokenizer.save_pretrained(exp_name)
    # Save the optimizer state
    torch.save(trainer.optimizer.state_dict(), os.path.join(exp_name, "optimizer.pt"))
    # Save the scheduler state
    torch.save(trainer.lr_scheduler.state_dict(), os.path.join(exp_name, "scheduler.pt"))
    upload_folder(exp_name)
    print("Model saved successfully.")
    
    
    # Extract the source (Text) and reference (Expected) texts
    # pred_texts = test_model(model, tokenizer, args.model_name,  test, args.max_seq_length)
    # log_and_save_results(test, pred_texts, exp_name=exp_name)
import argparse
      
if __name__ == "__main__":
    fourbit_models = {
        "llama-8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  
        "llama-8b-i": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "mistral-base": "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", 
        "mistral-i": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "mistral-7b": "unsloth/mistral-7b-v0.3-bnb-4bit",      
        "mistral-7b-i": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "gemma-2-9b": "unsloth/gemma-2-9b-bnb-4bit",
    }
    parser = argparse.ArgumentParser(description="Sinhala Spell Checker")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to be used")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Learning rate for training")
    parser.add_argument("--save_steps", type=int, default=100000, help="Number of steps to save the model")
    args = parser.parse_args()
    if args.model in fourbit_models:
        args.model_name = fourbit_models[args.model]
    else:
        print("Loading model from", args.model)
        args.model_name = args.model
    
    try:
    # exapmle: python unsloth_script.py --model llama-8b --exp_name exp_name --batch_size 4 --seed 42 --epochs 1 --gradient_accumulation_steps 2 --lr 1e-4
        main(args)
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        error_trace = traceback.format_exc()
        print(error_trace)
        mlflow.log_param("error", str(e))
        mlflow.log_param("status", "Failed")
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    mlflow.end_run()