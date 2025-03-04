import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Read csv into DataFrame
df = pd.read_csv('bbc_news.csv', nrows=500)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(train_size=0.8, seed=42)

# Convert back to Dataset-format
train_dataset = dataset["train"]
test_dataset = dataset["test"]

def tokenize_func(example):
    model_inputs = tokenizer(example["description"], max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(example["title"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_func, batched=True)
test_dataset = test_dataset.map(tokenize_func, batched=True)

training_args = TrainingArguments(
    output_dir="Einstiegsaufgabe_GenAI",
    evaluation_strategy ="epoch",
    per_gpu_train_batch_size=4,
    per_gpu_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")












