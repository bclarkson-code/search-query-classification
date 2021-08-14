import pickle
import torch
from tqdm.auto import tqdm
from gpt2_predictor import GPT2Predictor, GPT2TestSearchQueryDataModule

if __name__ == '__main__':
    encoding = {
        'Arts': 0,
        'Business': 11,
        'Computers': 10,
        'Games': 12,
        'Health': 9,
        'Home': 6,
        'News': 14,
        'Recreation': 1,
        'Reference': 13,
        'Regional': 4,
        'Science': 8,
        'Shopping': 3,
        'Society': 2,
        'Sports': 5,
        'World': 7
    }
    queries = GPT2TestSearchQueryDataModule(
        'open_source.feather',
        batch_size=128,
        num_workers=0,
        tokeniser_string='gpt2',
        debug=False,
        encoding=encoding,
    )
    queries.prepare_data()
    queries.setup()

    model = GPT2Predictor.load_from_checkpoint(
        'gpt2-checkpoints/model-epoch=00-valid/loss=1.86.ckpt',
        strict=False
    )
    test_data = queries.test_dataloader()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc='Predicting'):
            (input_ids, attention_mask), _ = batch
            pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds.append(pred)

    with open('test_preds.pkl', 'wb') as f:
        pickle.dump(preds)



