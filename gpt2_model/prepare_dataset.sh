gsutil -m cp gs://sqc/datasets/train.pkl .
gsutil -m cp gs://sqc/datasets/test.pkl .
gsutil -m cp gs://sqc/datasets/valid.pkl .
python3 prepare_dataset.py
