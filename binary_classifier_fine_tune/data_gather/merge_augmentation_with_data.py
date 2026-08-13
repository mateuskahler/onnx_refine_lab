import json
import csv

LANGUAGE_CODE = 'en'

AUGMENTED_DATA_FILENAME = "model_synthetic.jsonl"
ORGANIC_DATA_FILENAME = "merged_file.jsonl"
OUTPUT_FILENAME = "merged_file_with_augmentation.jsonl"

label_key = "is_it_about_ai"


with open(AUGMENTED_DATA_FILENAME, "r", encoding="utf-8") as augmented_file:
    def extract_as_dict(line):
        text, label = line
        if label == "yes":
            label_value = "yes"
        elif label == "no":
            label_value = "no"
        else:
            raise ValueError(f"Unexpected label value: {label}")

        return {"text": text, "language": LANGUAGE_CODE, label_key: label_value}
    

    comma_separated_reader = csv.reader(augmented_file, skipinitialspace=True)
    augmented_data = [extract_as_dict(line) for line in comma_separated_reader]

with open(ORGANIC_DATA_FILENAME, "r", encoding="utf-8") as organic_file:
    organic_data = [json.loads(line) for line in organic_file]


with open(OUTPUT_FILENAME, "w", encoding="utf-8") as output_file:
    for index, data in enumerate(organic_data):
        item_dict = {"id": f"o{index}", "language": data["language"], label_key: data[label_key], "is_it_organic":"yes", "text": data["text"]}
        output_file.write(json.dumps(item_dict) + "\n")
        
    for index, data in enumerate(augmented_data):
        item_dict = {"id": f"s{index}", "language": data["language"], label_key: data[label_key], "is_it_organic":"no", "text": data["text"]}
        output_file.write(json.dumps(item_dict) + "\n")

print(f"Merging complete! Output saved to: {OUTPUT_FILENAME}")
