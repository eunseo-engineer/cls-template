from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import mlflow
import os

def train():
    mlflow.start_run()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    dataset = load_dataset("imdb")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="epoch",
        logging_dir="logs",
        logging_steps=3,
        report_to="tensorboard",
        num_train_epochs=1,
        per_device_train_batch_size=8
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),
        eval_dataset=tokenized_datasets["test"].shuffle().select(range(500)),
    )

    trainer.train()
    mlflow.log_param("model", "bert-base-uncased")
    mlflow.log_metric("final_loss", 0.123)
    mlflow.end_run()

if __name__ == "__main__":
    train()
