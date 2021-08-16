from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import subprocess
import shlex
import os
import numpy as np

if __name__ == '__main__':
    Path('raw_dataset').mkdir(exist_ok=True)

    # Download the raw dataset from google cloud storage
    cmd = shlex.split(
        'gsutil -m cp gs://search-query-classification-us-central1-c/raw_dataset/* raw_dataset'
    )
    print(subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT).decode('utf-8')
    )

    df = pd.read_parquet('raw_dataset')

    # write the strings to a text file for the pretrain model
    text_ds_dir = 'text_datasets'
    Path(text_ds_dir).mkdir(exist_ok=True)
    train_frac = 0.8
    valid_frac = 0.1
    n_queries = len(df)
    datasets = np.split(
        df.sample(frac=1, random_state=42),
        [
            int(train_frac * n_queries),
            int((train_frac + valid_frac) * n_queries)
        ]
    )
    for ds, ds_name in zip(datasets, ['train', 'test', 'valid']):
        file_path = os.path.join(text_ds_dir, f'{ds_name}.txt')
        with open(file_path, 'w') as f:
            for query in tqdm(df['query_text'], dec=f'Writing {ds_name} queries to file'):
                f.write(query)
