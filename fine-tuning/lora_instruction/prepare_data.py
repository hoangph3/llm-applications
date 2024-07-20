
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
from torch.utils.data import random_split
import torch
import os
import ssl


ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
proxy = 'http://192.168.5.8:3128'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction,
       optional input and a 'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


def prepare_sample(example, tokenizer, masked_inputs, ignore_index):
    example = example.to_dict()
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example['response']
    encoded_full_prompt = tokenizer.encode(full_prompt)
    eos_id = tokenizer.eos_token
    encoded_full_prompt_and_response = torch.cat(
        [
            tokenizer.encode(full_prompt_and_response, return_tensors='pt').view(-1),
            tokenizer.encode(eos_id, return_tensors='pt', add_special_tokens=False).view(-1)
        ]
    )

    labels = encoded_full_prompt_and_response.clone()
    
    if masked_inputs:
        labels[:len(encoded_full_prompt)] = ignore_index
        
    return {
        **example,
        'input_ids': encoded_full_prompt_and_response,
        'input_ids_no_response': encoded_full_prompt,
        'labels': labels
    }

def prepare(
    test_size=0.1,
    destination_path=Path('data/dolly'),
    checkpoint_dir='mistralai/Mistral-7B-v0.1',
    seed=252
):
    destination_path.mkdir(parents=True, exist_ok=True)
    
    print('Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, token='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', cache_dir="../cache")
    
    data_file_url = 'https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl'
    data = pd.read_json(data_file_url, lines=True)
    data.columns = ['instruction', 'input', 'response', 'category']
    data = [example for _, example in data.iterrows()]
    
    train_set, test_set = random_split(
        data, 
        [1 - test_size, test_size], 
        torch.Generator().manual_seed(seed)
    )

    train_set, test_set = list(train_set), list(test_set)
    
    print(f'Train has {len(train_set)} samples')
    print(f'Test has {len(test_set)} samples')
    
    print('Processing train split ...')
    train_set = [
        prepare_sample(
            sample, 
            tokenizer=tokenizer,
            masked_inputs=False, 
            ignore_index=-1
        )
        for sample in tqdm(train_set)
    ]
    
    torch.save(train_set, destination_path / 'train_set.pt')
    
    print('Processing test split ...')
    test_set = [
        prepare_sample(
            sample, 
            tokenizer=tokenizer,
            masked_inputs=False, 
            ignore_index=-1
        )
        for sample in tqdm(test_set)
    ]
    
    torch.save(test_set, destination_path / 'test_set.pt')

