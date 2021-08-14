import torch
import os
import pandas as pd
from gpt2_predictor import GPT2Predictor
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from pytorch_lightning import Trainer

class TokenDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset that supplies tokenised query
    inputs to the embedder
    """
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        input_ids = self.tokens['input_ids'][idx]
        attention_mask = self.tokens['attention_mask'][idx]
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

def split_query(query):
    """
    Generate a list of all of the partial
    queries for a given complete query
    """
    return [query[:i] for i in range(3, len(query) + 1)]

def collect_input(row):
    """
    Build the model input for a single query
    """
    global inputs
    for split in split_query(row['Query']):
        input_dict = {key: row[key] for key in ['historical_embedding', 'historical_label', 'category']}
        input_dict['Query'] = split
        inputs.append(input_dict)

def tokenize_function(strings, tokeniser):
    return tokeniser(
        strings,
        padding='max_length',
        truncation=True,
        max_length=24,
    )

def build_inputs(df):
    """
    Construct an input dataframe with embeddings
    and split query strings
    """
    global inputs
    inputs = []
    df.apply(collect_input, axis=1)
    return pd.DataFrame(inputs)

if __name__ == '__main__':
    # Load Embedder
    embedder = GPT2Predictor.load_from_checkpoint(
        'gpt2-checkpoints/model-epoch=00-valid/loss=1.86.ckpt',
        strict=False
    )

    # Build tokeniser
    tokeniser = AutoTokenizer.from_pretrained('gpt2')
    tokeniser.pad_token = tokeniser.eos_token

    for ds_name in tqdm(['train', 'valid', 'test'], desc='Embedding'):
        # Read the dataset
        print('Reading dataset')
        base_df = pd.read_feather(f'datasets/aol_data_{ds_name}.feather')
        print('Building inputs')
        input_df = build_inputs(base_df)

        # Tokenise the queries
        print('Tokenising')
        tokens = tokenize_function(input_df['Query'].tolist(), tokeniser)
        token_ds = TokenDataset(tokens)
        token_loader = DataLoader(token_ds, batch_size=512, num_workers=os.cpu_count())

        # Generate embeddings
        trainer = Trainer(
            tpu_cores=8,
            progress_bar_refresh_rate=1
        )
        with torch.no_grad():
            preds = trainer.predict(embedder, token_loader)

        preds = np.concatenate(preds)
        input_df['query_embedding'] = preds.tolist()
        input_df.to_feather(f'datasets/aol_data_{ds_name}_input_df.feather')
