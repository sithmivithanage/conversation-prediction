from unsloth import FastLanguageModel
from SinhalaSpellChecker.util_classes.data_processing import read_and_clean_data
from datasets import Dataset
import torch
import mlflow

alpaca_prompt = """You are an Expert Sinhala Spell Corrector. Below is a sentence in Sinhala Language. It may or may not have a spelling mistake. Give the corrected output in Sinhala.

### Text:
{}

### Output:
{}"""

def get_model(model_name, max_seq_length, dtype, load_in_4bit, seed, attn_implementation="flash_attention_2"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        attn_implementation=attn_implementation,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = seed,
        use_rslora = False,  # support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer
   
def test_fast_model(model, tokenizer, src_texts, batch_size, max_seq_length=2048):
    from tqdm import tqdm
    from itertools import islice
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
        print("Padding side changed to left")
    tokenizer.pad_token = tokenizer.eos_token
    pred_texts = []  
    FastLanguageModel.for_inference(model) 
    def chunked_iterable(iterable, batch_size):
        it = iter(iterable)
        return iter(lambda: list(islice(it, batch_size)), [])
    
    batched_src_texts = chunked_iterable(src_texts, batch_size)
    for batch in tqdm(batched_src_texts, desc="Generating predictions"):
        # Format prompts for the batch
        batched_prompts = [alpaca_prompt.format(src, "") for src in batch]
        
        # Tokenize the batch
        inputs = tokenizer(batched_prompts, return_tensors="pt", padding='longest').to("cuda")
        # print(inputs['input_ids'].shape)
        # Generate outputs for the batch
        outputs = model.generate(**inputs, max_new_tokens=max_seq_length, use_cache=True)
        # print(outputs)
        batch_pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_texts.extend(batch_pred_texts)  # Add all predictions to pred_texts 
    
    return pred_texts

def test_model(model, tokenizer, model_path, src_texts, max_seq_length, do_sample = True, method="no_mode"):
    from tqdm import tqdm
    
    if model is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="lora_model",
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    FastLanguageModel.for_inference(model) 
    pred_texts = []
    generation_mode = model.generation_config.get_generation_mode()
    print(f"Generation mode: {generation_mode}")
    if method=="no_mode":
        for src in tqdm(src_texts, desc="Generating predictions"):
            inputs = tokenizer(
                [alpaca_prompt.format(src, "")],
                return_tensors="pt"
            ).to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=max_seq_length, do_sample=do_sample, use_cache=True)

            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_texts.append(pred_text)
    elif method=="temp":
        print("Using Custom Temperature Sampling")
        for src in tqdm(src_texts, desc="Generating predictions"):
            inputs = tokenizer(
                [alpaca_prompt.format(src, "")],
                return_tensors="pt"
            ).to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=max_seq_length, use_cache=True, temperature=0.1)

            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_texts.append(pred_text)
    elif method == "bad_words":
        from transformers import AutoTokenizer
        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        def get_tokens_as_list(word_list):
            "Converts a sequence of words into a list of tokens"
            tokens_list = []
            for word in word_list:
                tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
                tokens_list.append(tokenized_word)
            return tokens_list
        bad_words_ids = get_tokens_as_list(word_list=["###", "Text:", "Output:"])
        print("Bad Words", bad_words_ids)
        
        for src in tqdm(src_texts, desc="Generating predictions"):
            inputs = tokenizer(
                [alpaca_prompt.format(src, "")],
                return_tensors="pt"
            ).to("cuda")

            outputs = model.generate(**inputs, bad_words_ids=bad_words_ids, max_new_tokens=max_seq_length, use_cache=True)

            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_texts.append(pred_text)
    else:
        print("Invalid Method Argument")
    return pred_texts

def prepare_datasets_with_formatting(EOS_TOKEN):
    train, val, test = read_and_clean_data(dataset_size=0.02)

    def formatting_prompts_func(examples):
        text = examples["text"]
        expected = examples["expected"]
        texts = []
        for text, expected in zip(text, expected):
            formatted_text = alpaca_prompt.format(text, expected) + EOS_TOKEN
            # formatted_text = f"Input: {text}\nExpected Output: {expected}" + EOS_TOKEN
            texts.append(formatted_text)
        return { "text" : texts }

    train_dataset = Dataset.from_pandas(train)
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True, )

    val_dataset = Dataset.from_pandas(val)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
    return train_dataset, val_dataset, test

def prepare_datasets(EOS_TOKEN):
    # TODO : Add dataset size parameter
    train, val, test = read_and_clean_data(dataset_size=0.1)
    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    return train_dataset, val_dataset, test

import re

def clean_text(text):
    match = re.search(r'### Output:\n(.*)', text, re.DOTALL)
    if match:
        return match.group(1).replace("Text:\n", "").replace("Output:\n", "").replace('###', '').strip()
    return text.strip()

def log_and_save_results(test_set, pred_texts, exp_name="Unsloth Test", upload_folder_id=None):
    import pandas as pd
    from SinhalaSpellChecker.utils.nlp import evalute
    from SinhalaSpellChecker.utils.google import upload_file
    from SinhalaSpellChecker.utils.general import print_in_tab_seperated_format

    src_texts = test_set['text'].tolist()
    ref_texts = test_set['expected'].tolist()
    out_texts = [clean_text(text) for text in pred_texts]
    print(len(src_texts), len(ref_texts), len(pred_texts))
        
    df = pd.DataFrame({
         'Text': src_texts,
         'Output': out_texts,
         'Expected': ref_texts,
         '_Output': pred_texts,
    })

    df.to_csv('output.csv', index=False)
    df.to_csv(f'Predictions-{exp_name}.csv')
    df.to_excel(f'Predictions-{exp_name}.xlsx')
    corrected_pred_texts = df['Output'].tolist()
    eval_results = evalute(srcs=src_texts, preds=corrected_pred_texts, refs=ref_texts, test_name=exp_name)
    eval_with_replace = evalute(srcs=src_texts, preds=corrected_pred_texts, refs=ref_texts, test_name=exp_name, replace=True)
    with open(f'Results-{exp_name}.txt', 'w') as log_file:
        for metric, value in eval_results.items():
            log_file.write(f"{metric}: {value}\n")
            # mlflow.log_metric(metric, value)
    with open(f'Results-{exp_name}-replace.txt', 'w') as log_file:
        for metric, value in eval_with_replace.items():
            log_file.write(f"{metric}: {value}\n")
    
    df_eval = pd.DataFrame({
        'Metric': eval_with_replace.keys(),
        'Without Replacement': eval_results.values(),
        'With Replacement': [eval_with_replace[metric] for metric in eval_with_replace.keys()]
    })
    
    print("------------------------------------------------------------------------------------------------------------")
    print(df_eval.to_string(index=False))
    print("------------------------------------------------------------------------------------------------------------")
    if upload_folder_id is not None:
        upload_file(f'Predictions-{exp_name}.csv', f'Predictions-{exp_name}.csv', upload_folder_id)
        upload_file(f'Predictions-{exp_name}.xlsx', f'Predictions-{exp_name}.xlsx', upload_folder_id)
        upload_file(f'Results-{exp_name}.txt', f'Results-{exp_name}.txt', upload_folder_id)
        upload_file(f'Results-{exp_name}-replace.txt', f'Results-{exp_name}-replace.txt', upload_folder_id)
    
    print_in_tab_seperated_format(df_eval)