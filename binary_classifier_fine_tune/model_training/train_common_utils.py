
import random
from typing import Iterator, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler
from transformers import AutoConfig, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from collections import Counter

STATE_SEED = 85


####################################################
# Loaders

def process_raw_line_as_sample(item: dict[str, Any]) -> dict[str, Any]:
    """Extracts and normalizes fields from a raw JSON record."""
    return {
        "id": item["id"],
        "language": item["language"],
        "organic": True if item.get("is_it_organic") == "yes" else False,
        "label_id": 0 if item.get("is_it_about_ai") == "no" else 1,
        "text": item["text"],
    }


def prepare_dataset(
        raw_json_data: list[dict[str, Any]],
        train_test_split_ratio: float = 0.05,
        filter_languages: list[str] = ["en"]) -> DatasetDict:
    processed_records = []
    for entry in raw_json_data:
        record = process_raw_line_as_sample(entry)
        if record["language"] in filter_languages:
            processed_records.append(record)

    organic_samples = [r for r in processed_records if r["organic"]]
    synthetic_samples = [r for r in processed_records if not r["organic"]]

    # Helper function to print label proportions
    def print_distribution(name: str, records: list[dict[str, Any]]):
        total = len(records)
        if total == 0:
            print(f"[{name}] Empty set")
            return
        counts = Counter(r["label_id"] for r in records)
        prop_0 = (counts[0] / total) * 100
        prop_1 = (counts[1] / total) * 100
        print(f"[{name}] Total: {total} | Label 0: {counts[0]} ({prop_0:.1f}%) | Label 1: {counts[1]} ({prop_1:.1f}%)")

    print("=== Raw Class Distributions ===")
    print_distribution("Organic Total", organic_samples)
    print_distribution("Synthetic Total", synthetic_samples)
    print("-" * 50)

    # Stratify organic data by label_id
    org_train, org_test = train_test_split(
        organic_samples,
        test_size=train_test_split_ratio,
        random_state=STATE_SEED,
        stratify=[r["label_id"] for r in organic_samples],
    )

    # Stratify synthetic data by label_id
    syn_train, syn_test = train_test_split(
        synthetic_samples,
        test_size=train_test_split_ratio,
        random_state=STATE_SEED,
        stratify=[r["label_id"] for r in synthetic_samples],
    )

    # Combine subsets
    train_data = org_train + syn_train
    test_data = org_test + syn_test

    print("=== Final Split Distributions ===")
    print_distribution("Organic Train", org_train)
    print_distribution("Organic Test ", org_test)
    print_distribution("Synthetic Train", syn_train)
    print_distribution("Synthetic Test ", syn_test)
    print("-" * 50)

    # Structuring as Hugging Face Datasets
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })

    return dataset_dict

def load_tokenizer_and_model(model_name: str, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config)
    return tokenizer, model


#####################################

class AlternatingClassSampler(Sampler[int]):
    """
    Custom PyTorch Sampler that interleaves class indices (0, 1, 0, 1...).
    Recycles indices from the minority class until the majority class is exhausted.
    """

    def __init__(self, labels: list[int], shuffle: bool = True):
        self.labels = labels
        self.shuffle = shuffle

        # Group indices by target class
        self.class_0_indices = [idx for idx,
                                label in enumerate(labels) if label == 0]
        self.class_1_indices = [idx for idx,
                                label in enumerate(labels) if label == 1]

        # Calculate epoch length based on the majority class
        self.max_class_len = max(
            len(self.class_0_indices), len(self.class_1_indices))
        self.total_length = self.max_class_len * 2

    def __iter__(self) -> Iterator[int]:
        c0 = self.class_0_indices.copy()
        c1 = self.class_1_indices.copy()

        if self.shuffle:
            random.shuffle(c0)
            random.shuffle(c1)

        len0, len1 = len(c0), len(c1)

        # Interleave samples
        for i in range(self.max_class_len):
            yield c0[i % len0]  # Class 0 sample
            yield c1[i % len1]  # Class 1 sample

    def __len__(self) -> int:
        return self.total_length


class AlternatingTrainer(Trainer):
    """
    Subclass of Hugging Face Trainer that forces the training DataLoader
    to use AlternatingClassSampler instead of the default random sampler.
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        labels = self.train_dataset["label_id"]
        train_sampler = AlternatingClassSampler(labels=labels, shuffle=True)

        return DataLoader(
            self.train_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )



def prepare_tokenized_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 512) -> DatasetDict:
    """Tokenizes text and strips un-tensorable raw string columns."""

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    column_names = ds_dict["train"].column_names

    return ds_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=[col for col in column_names if col not in [
            "label_id", "organic"]]
    )
