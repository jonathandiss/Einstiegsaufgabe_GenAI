import torch
import pandas as pd
from datasets import load_dataset, Dataset
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer


# Check for cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model.to(device)

# Read CSV as DataFrame
df = pd.read_csv('bbc_news.csv', nrows=500)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Split Dataset
dataset = dataset.train_test_split(train_size=0.8, seed=42)

# Convert back to Dataset-format
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Tokenize Dataset
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['description'], max_length=1024, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=examples['title'], max_length=64, truncation=True, padding="max_length"
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Batching function
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Arguments of the training
training_args = transformers.Seq2SeqTrainingArguments(
    output_dir='trained_model',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    disable_tqdm=not torch.cuda.is_available()
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
