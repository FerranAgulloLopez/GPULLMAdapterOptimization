import json
import os

DATASET_PATH = '../data/ShareGPT_V3_unfiltered_cleaned_split.json'
NEW_DATASET_PATH = '../data/ShareGPT_V3_unfiltered_cleaned_split_with_expected_output_lens_reduced.json'
OUTPUT_LENS_PATH = './output/output_lens.json'


def main():
    global DATASET_PATH, NEW_DATASET_PATH, OUTPUT_LENS_PATH

    with open(DATASET_PATH) as json_file:
        dataset = json.load(json_file)

    with open(OUTPUT_LENS_PATH) as json_file:
        output_lens = dict(json.load(json_file))

    warnings = 0
    new_dataset = []
    for index_conversation, conversation in enumerate(dataset):
        conversation_id = conversation['id']
        if len(conversation['conversations']) > 0:
            index_item = 0
            if conversation_id not in output_lens:
                warnings += 1
            else:
                dataset[index_conversation]['conversations'][index_item]['expected_output_len'] = output_lens[conversation_id]
                new_dataset.append(dataset[index_conversation])

    with open(os.path.join(NEW_DATASET_PATH), 'w', encoding='utf8') as json_file:
        json.dump(new_dataset, json_file, ensure_ascii=True, indent=4)

    print(f'Warnings: {warnings}')


if __name__ == "__main__":
    main()
