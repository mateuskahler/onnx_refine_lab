import json

# which fields to pull from reference file
FIELDS_TO_BRING = ["language", "is_it_about_ai"]

training_data_filename = "training_data.jsonl"
reference_answers_filename = "model_responses.jsonl"
output_filename = "merged_file.jsonl"

reference_answers_dict = {}

with open(reference_answers_filename, "r", encoding="utf-8") as f2:
    for line in f2:
        if line.strip():
            record2 = json.loads(line)
            item_id = record2.get("id")
            if item_id is not None:
                reference_answers_dict[item_id] = record2

with open(training_data_filename, "r", encoding="utf-8") as f_in, \
     open(output_filename, "w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue
            
        record1 = json.loads(line)
        item_id = record1.get("id")
        
        try:
            record2 = reference_answers_dict[item_id]

            for field in FIELDS_TO_BRING:
                if field in record2:
                    record1[field] = record2[field]
            
            f_out.write(json.dumps(record1, ensure_ascii=False) + "\n")

        except KeyError:
            print(f"Warning: ID {item_id} not found in reference answers.")
            continue

print("Merging complete! Output saved to:", output_filename)
