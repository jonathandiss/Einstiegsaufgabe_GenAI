import pandas as pd
from datasets import load_dataset, Dataset

df = pd.read_csv('bbc_news.csv', nrows=500)
dataset = Dataset.from_pandas(df)
print(dataset)