import os
import json
import matplotlib.pyplot as plt
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import Tuple, List
import numpy as np

MODEL_PATH = '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b'
DATASET_PATH = '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/ShareGPT_V3_unfiltered_cleaned_split.json'
NEW_DATASET_PATH = '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data'
OUTPUT_PATH = '../output'


class DatasetCharacteristics:
    def __init__(
            self,
            prompt_lens_mean: float,
            prompt_lens_p25: float,
            prompt_lens_p75: float,
            output_lens_mean: float,
            output_lens_p25: float,
            output_lens_p75: float,
    ):
        self.prompt_lens_mean = prompt_lens_mean
        self.prompt_lens_p25 = prompt_lens_p25
        self.prompt_lens_p75 = prompt_lens_p75
        self.output_lens_mean = output_lens_mean
        self.output_lens_p25 = output_lens_p25
        self.output_lens_p75 = output_lens_p75

    def __str__(self):
        return f'Prompt Lens -> Mean {self.prompt_lens_mean}; P25 {self.prompt_lens_p25}; P75 {self.prompt_lens_p75}\n' \
               f'Output Lens -> Mean {self.output_lens_mean}; P25 {self.output_lens_p25}; P75 {self.output_lens_p75}'


def find_dataset_characteristics() -> DatasetCharacteristics:
    global MODEL_PATH, DATASET_PATH, OUTPUT_PATH

    tokenizer = get_tokenizer(MODEL_PATH, trust_remote_code=True)

    with open(DATASET_PATH) as json_file:
        dataset = json.load(json_file)

    ids: List[str] = []
    prompt_lens: List[int] = []
    output_lens: List[int] = []
    for index_conversation, conversation in enumerate(dataset):
        if len(conversation['conversations']) >= 2:
            conversation_id = conversation['id']
            prompt = conversation['conversations'][0]['value']
            prompt_len = len(tokenizer(prompt).input_ids)
            output = conversation['conversations'][1]['value']
            output_len = len(tokenizer(output).input_ids)
            if prompt_len < 4 or output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            ids.append(conversation_id)
            prompt_lens.append(prompt_len)
            output_lens.append(output_len)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(prompt_lens, bins=30)
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.title('Prompt lengths')
    plt.savefig(os.path.join(OUTPUT_PATH, 'prompt_lengths'), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(output_lens, bins=30)
    plt.xlabel('length')
    plt.ylabel('frequency')
    plt.title('Output lengths')
    plt.savefig(os.path.join(OUTPUT_PATH, 'output_lengths'), bbox_inches='tight')

    output = DatasetCharacteristics(
        prompt_lens_mean=np.mean(prompt_lens),
        prompt_lens_p25=np.percentile(prompt_lens, 25),
        prompt_lens_p75=np.percentile(prompt_lens, 75),
        output_lens_mean=np.mean(output_lens),
        output_lens_p25=np.percentile(output_lens, 25),
        output_lens_p75=np.percentile(output_lens, 75),
    )

    print('Dataset characteristics', output)

    return output


def find_lens_in_range(chosen_prompt_lens: float, chosen_output_lens: float) -> str:
    global MODEL_PATH, DATASET_PATH, OUTPUT_PATH

    tokenizer = get_tokenizer(MODEL_PATH, trust_remote_code=True)

    with open(DATASET_PATH) as json_file:
        dataset = json.load(json_file)

    min_distance: float = None
    min_id: str = None
    min_prompt_len: int = None
    min_output_len: int = None
    for index_conversation, conversation in enumerate(dataset):
        if len(conversation['conversations']) >= 2:
            conversation_id = conversation['id']
            prompt = conversation['conversations'][0]['value']
            prompt_len = len(tokenizer(prompt).input_ids)
            output = conversation['conversations'][1]['value']
            output_len = len(tokenizer(output).input_ids)
            if prompt_len < 4 or output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue

            distance = abs(chosen_prompt_lens - prompt_len) + abs(chosen_output_lens - output_len)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_id = conversation_id
                min_prompt_len = prompt_len
                min_output_len = output_len

    print(f'Minimum distance to means -> Id: {min_id}; prompt length: {min_prompt_len}; output_length: {min_output_len}')

    return min_id


def create_dataset(selected_id: str, title: str):
    global MODEL_PATH, DATASET_PATH, OUTPUT_PATH, NEW_DATASET_PATH

    with open(DATASET_PATH) as json_file:
        dataset = json.load(json_file)

    selected_conversation = None
    for index_conversation, conversation in enumerate(dataset):
        if conversation['id'] == selected_id:
            selected_conversation = {
                'id': selected_id,
                'conversations': [
                    conversation['conversations'][0],
                    conversation['conversations'][1]
                ]
            }
            break

    if selected_conversation is None:
        raise ValueError('Selected conversation should exist in the dataset')

    new_dataset = [selected_conversation] * len(dataset)

    with open(os.path.join(NEW_DATASET_PATH, f'dummy_dataset_{title}.json'), 'w', encoding='utf8') as json_file:
        json.dump(new_dataset, json_file, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    dataset_characteristics = find_dataset_characteristics()
    selected_id = find_lens_in_range(dataset_characteristics.prompt_lens_mean, dataset_characteristics.output_lens_mean)
    create_dataset(selected_id, '_mean')
    selected_id = find_lens_in_range(dataset_characteristics.prompt_lens_p25, dataset_characteristics.output_lens_p25)
    create_dataset(selected_id, '_p25')
    selected_id = find_lens_in_range(dataset_characteristics.prompt_lens_p75, dataset_characteristics.output_lens_p75)
    create_dataset(selected_id, '_p75')
