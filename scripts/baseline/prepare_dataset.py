from datasets import load_dataset
import pandas as pd
import os

if not os.path.exists('./data/prepared_data'):
    os.makedirs('./data/prepared_data')

dataset = load_dataset('iwslt2017', pair='de-en', is_multilingual = False,  cache_dir='./data')
df = pd.DataFrame(dataset['train']['translation'], columns=['de', 'en'])
df.to_csv('./data/prepared_data/train_de-en.csv', index=False)

df = pd.DataFrame(dataset['validation']['translation'], columns=['de', 'en'])
df.to_csv('./data/prepared_data/val_de-en.csv', index=False)

df = pd.DataFrame(dataset['test']['translation'], columns=['de', 'en'])
df.to_csv('./data/prepared_data/test_de-en.csv', index=False)