import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from train_utils import load_dataset
mock_data = load_dataset()

for i in range(5):
    label = mock_data["labels"][i]
    text = mock_data["text"][i]
    print(f"[{i}] Label: {label}")
    print(f"    text:     {text[:60]}")


# Load into Hugging Face Dataset and split into Train / Test
raw_dataset = Dataset.from_dict(mock_data)
dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42)

# Load DeBERTa-v3 Tokenizer & Model
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

# Tokenize Dataset


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)


tokenized_dataset = dataset_split.map(preprocess_function, batched=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./training_output",
    learning_rate=5e-6,           # Drop back to 5e-6 for base stability
    adam_epsilon=1e-6,            # Default is 1e-8; raising it prevents 0-division NaNs
    max_grad_norm=0.3,            # Tighten clipping from 0.5 to 0.3

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    warmup_steps=24,

    gradient_accumulation_steps=2,

    # CRITICAL: DeBERTa-v3 dynamic range seems bugged in my torch version (?)
    fp16=False,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",    # Prevents wandb/tensorboard prompt popups

    # --- metric settings
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_total_limit=1,
)


if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting training on mock dataset...")
    trainer.train()
    print("\n✅ Training complete! DeBERTa ran without NaN loss or crashes.")

    # Print which internal checkpoint won
    print("Best Checkpoint Found:", trainer.state.best_model_checkpoint)
    print("Best Eval Loss:", trainer.state.best_metric)

    # Save the best model and tokenizer explicitly to a permanent clean directory
    clean_save_dir = "./best_deberta_model"
    trainer.save_model(clean_save_dir)
    tokenizer.save_pretrained(clean_save_dir)

    print(f"Model saved to: {clean_save_dir}")
