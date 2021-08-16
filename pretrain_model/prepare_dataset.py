from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import subprocess
import shlex

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
    with open('raw_queries.txt', 'w') as f:
        for query in tqdm(df['query_text']):
            f.write(query)