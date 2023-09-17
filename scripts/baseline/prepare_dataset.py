from datasets import load_dataset
import pandas as pd
import os

if not os.path.exists('./data/prepared_data'):
    os.makedirs('./data/prepared_data')

subset_of_data = 'de-en' # options: 'en-fr', 'fr-en', 'en-de', 'de-en' ...
src_lang, trg_lang = subset_of_data.split('-')
target_subset = src_lang + '_' + trg_lang
dataset = load_dataset('iwslt2017', pair= subset_of_data, is_multilingual = False,  cache_dir='./data')
df = pd.DataFrame(dataset['train']['translation'], columns=[src_lang, trg_lang])
df.to_csv(f'./data/prepared_data/train_{target_subset}.csv', index=False)

df = pd.DataFrame(dataset['validation']['translation'], columns=[src_lang, trg_lang])
df.to_csv(f'./data/prepared_data/val_{target_subset}.csv', index=False)

df = pd.DataFrame(dataset['test']['translation'], columns=[src_lang, trg_lang])
df.to_csv(f'./data/prepared_data/test_{target_subset}.csv', index=False)