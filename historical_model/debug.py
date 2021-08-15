import pandas as pd
import numpy as np
from glob import glob
import pickle
from tqdm.auto import tqdm
from pathlib import Path

ds_name = 'train'

print('Reading Predictions')
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

raw_input_dir = 'raw_inputs'
Path(raw_input_dir).mkdir(exist_ok=True)
raw_input_path = f'{raw_input_dir}/{ds_name}.pkl'
input_df = pd.read_pickle(raw_input_path)

df = pd.concat([input_df.reset_index(drop=True), pd.DataFrame(preds).reset_index(drop=True)],
               axis=1)




