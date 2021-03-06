import torch
import os
import pandas as pd
from gpt2_predictor import GPT2Predictor
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from pytorch_lightning import Trainer
from pathlib import Path
import pickle
from pytorch_lightning.callbacks import BasePredictionWriter
from glob import glob

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

class PredWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval='batch'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True)

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
    ):
        save_path = os.path.join(
            self.output_dir,
            f"{str(batch_idx).zfill(5)}.pkl")
        prediction = prediction.cpu().numpy()
        with open(save_path, 'wb') as f:
            pickle.dump(prediction, f)
        del prediction

    def write_on_epoch_end(
            self,
            trainer,
            pl_module,
            predictions,
            batch_indices
    ):
        predictions = [[pred.cpu() for pred in output] for output in predictions]
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))

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
        return_tensors='pt',
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
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


    for ds_name in tqdm(['valid', 'train', 'test'], desc='Embedding'):
        # Either get input_df from disk or build it
        raw_input_dir = 'raw_inputs'
        Path(raw_input_dir).mkdir(exist_ok=True)
        raw_input_path = f'{raw_input_dir}/{ds_name}.pkl'

        if not os.path.exists(raw_input_path):
            # Read the dataset
            print('Reading dataset')
            base_df = pd.read_feather(f'datasets/aol_data_{ds_name}.feather')

            # Build inputs
            print('Building inputs')
            input_df = build_inputs(base_df)
            input_df.to_pickle(raw_input_path)
        else:
            print('Reading inputs')
            input_df = pd.read_pickle(raw_input_path)
        if not os.path.exists(f'{ds_name}_preds'):
            # Either get tokens from disk or build it
            token_file_path = f'tokens/{ds_name}.pkl'
            Path('tokens').mkdir(exist_ok=True)
            if not os.path.exists(token_file_path):
                # Tokenise the queries
                print('Tokenising')
                tokens = tokenize_function(input_df['Query'].tolist(), tokeniser)
                with open(token_file_path, 'wb') as f:
                    pickle.dump(tokens, f)
            else:
                print('Reading tokens')
                with open(token_file_path, 'rb') as f:
                    tokens = pickle.load(f)

            token_ds = TokenDataset(tokens)
            token_loader = DataLoader(
                token_ds,
                batch_size=128,
                num_workers=os.cpu_count()
            )
            pred_writer = PredWriter(output_dir=f'{ds_name}_preds')
            # Generate embeddings
            trainer = Trainer(
                tpu_cores=1,
                progress_bar_refresh_rate=1,
                accelerator='ddp_spawn',
                callbacks=pred_writer
            )
            with torch.no_grad():
                trainer.predict(embedder, token_loader)

        print('Reading predictions')
        pred_dir = f'{ds_name}_preds'
        pred_paths = glob(pred_dir + '/*')
        pred_paths = sorted(pred_paths)

        preds = []
        for path in pred_paths:
            with open(path, 'rb') as f:
                pred = pickle.load(f)
                preds.append(pred)

        print('Saving predictions')
        preds = np.concatenate(preds)
        df = pd.concat([
            input_df.reset_index(drop=True),
            pd.DataFrame(preds).reset_index(drop=True)
        ], axis=1)
        df.to_pickle(f'datasets/aol_data_{ds_name}_input_df.pkl')
