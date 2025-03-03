import kagglehub
import pandas as pd
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Read csv into DataFrame
df = pd.read_csv('data/bbc_news.csv', nrows=2000)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

train_texts, test_texts, train_labels, test_labels = dataset.train_test_split(train_size=0.8, seed=42)

# Convert back to Dataset-format
train_dataset = Dataset.from_dict({"description": train_texts, "title": train_labels})
test_dataset = Dataset.from_dict({"description": test_texts, "title": test_labels})

def tokenize_func(example):
    model_inputs = tokenizer(example["description"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["title"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_func, batched=True)
test_dataset = test_dataset.map(tokenize_func, batched=True)

training_args = TrainingArguments(
    output_dir="Einstiegsaufgabe_GenAI",
    evaluation_strategy ="epoch",
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)














