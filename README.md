# Einstiegsaufgabe_GenAI

Ziel dieser Einstiegsaufgabe ist es, den Vorgang zu entwickeln, mit dem zu gegebenen Nachrichtentexten jeweils Überschriften generiert werden. <br>
<br>
Das Projekt verwendet das vortrainiertes Modell ```facebook/bart-large-cnn``` und einen Datensatz von BBC-News. <br>
Das trainerte Modell ist aufgrund der Größe nicht Bestandteil des Repositorys.

## Skripte
***
In der Aufgabe wurden folgende Skripte erstellt:
*  [training.py](training.py): Training des Modells
*  [test_model.py](test_model.py): Testen des Modells
* [analyse_model.py](analyse_dataset.py): Analyse des Datensatzes

Auf Letzteres wird nicht weiter eingegangen, da es ausschließlich dazu diente den Datensatz auf seine Struktur zu untersuchen.<br>

## [training.py](training.py)
*Hinweis: Die Bezeichnung von Test-Datensatz beschreibt den Datensatz für die Validierung.
Wegen der Bezeichnung aus der train_test_split()-Methode wurde dieser Name beibehalten.*
### 1. Import der Bibliotheken
````python 
import torch
import pandas as pd
from datasets import load_dataset, Dataset
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
````

### 2. Prüfen, ob GPU verfügbar ist
````python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")
````

### 3. Angabe des Tokenizers und des Modells
````python
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model.to(device)
````

### 4. Laden des Datensatzes
Aufgrund hoher Rechenleistung werden nur 500 Zeilen aus der CSV geladen und verarbeitet. Der Datensatz wird in 80% Training und 20% Test aufgeteilt.
````python
df = pd.read_csv('bbc_news.csv', nrows=500)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(train_size=0.8, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
````

### 5. Preprocessing Funktion
Beschreibungen gelten als Eingabe und werden vom Tokenizer in IDs verarbeitet.<br>
Titel gelten als Laben und werden auch vom Tokenizer in IDs verarbeitet. Im Gegensatz zu der Eingabe werden Informationen wie z.B. attention_mask hier nicht benötigt.<br>
Anschließend wird die Funktion auf mehrere Elemente aus dem Dataset gleichzeitig angewendet.
````python
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
````

### 6. Trainingsparameter und Training
Ein Daten-Collator unterstützt die Batches mit dynamischen Padding, indem er die Padding-Länge an die längste Sequenz im Batch anpasst. Daher könnte theoretisch auf den Parameter "max_length" in der Preprocessing Funktion verzichtet werden.<br>
#### Trainingsparameter:
- evaluation_strategy: Modell wird nach jeder Epoche evaluiert
- learning_rate: kleine Lernrate
- batch_size: Batch-Größe für das Training und Evaluation
- weight_decay: Gewichtsanpassungen (Overfitting reduzieren)
- save_total_limit: Maximal drei Checkpoints werden gespeichert
- num_train_epochs: Das Modell trainiert 10 Epochen
- fp16: Beschleunigt Berechnungen bei GPU
- predict_with_generate: Nutzt generate()-Methode für die Vorhersage
- disable_tqdm: Deaktiviert Fortschrittsbalken wenn CPU genutzt wird

#### Trainer:
- model: Gibt das zu trainierende Modell an
- args: Verwendet die angegebenen Trainingsparameter
- dataset: Verwendet den angegebenen Datensatz
- tokenizer: Gibt den Tokenizer an
- data_collator: Dynamisches Padding beim Training
````python
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

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
    disable_tqdm=not torch.cuda.is_available())

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,)

trainer.train()
````
## [test_model.py](test_model.py)
### 1. Angabe des Tokenizers und des Modells
````python
model_path = "trained_model/checkpoint-100"
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained(model_path)
````
### 2. Input-Text
````python
text = ('Hier steht ein Text')
````
### 3. Verarbeitung des Inputs in Token-IDs
Wenn der Text zu lang ist, wird er auf die maximale Länge gekürzt. Der Output ist in Form eines Pytorch Tensors.
````python
input_ids = tokenizer.encode(
    text,
    return_tensors="pt",
    max_length=1024,
    truncation=True)
````
### 4. Generierung von Token-IDs auf Grundlage der Input-IDs
Der Output betrachtet vier Token-Sequenzen und wählt die beste aus. Er ist in seiner Länge begrenzt.
````python
summary_text_ids = model.generate(
    input_ids=input_ids,
    max_length=142,
    min_length=56,
    num_beams=4)
````
### 5. Verarbeitung der Output-IDs in Strings und Ausgabe
Die generierten IDs werden in Strings umgewandelt. Dabei wird der erste Index genutzt, da num_beams die beste Sequenz als erste Ausgabe zurückgibt.<br>
Special Token wie z.B. die ganzen Paddings werden ignoriert.
````python
decoded_text = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
print(decoded_text)
````
## Mögliche Next Steps:
- Aufteilung des Datensatzes in Train, Test und Evaluation
- Größere Dimensionen im Training (größerer Datensatz, mehr Epochen, etc.)
- Anzeige von Metriken (loss, etc.) während dem Training
- Metriken für die Bewertung des Modells nach dem Training