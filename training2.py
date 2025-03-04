import pandas as pd
from datasets import load_dataset, Dataset
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer

df = pd.read_csv('bbc_news.csv', nrows=500)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(train_size=0.8, seed=42)

# Convert back to Dataset-format
train_dataset = dataset["train"]
test_dataset = dataset["test"]

tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['description'], max_length=1024, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=examples['title'], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
# Batching function
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define arguments of the finetuning
training_args = transformers.Seq2SeqTrainingArguments(
    output_dir='trained_model2',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # batch size for train
    per_device_eval_batch_size=4,  # batch size for eval
    weight_decay=.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
    predict_with_generate=True
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
