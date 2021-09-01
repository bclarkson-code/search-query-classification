gsutil -m cp gs://sqc/datasets/train.pkl datasets/train.pkl
gsutil -m cp gs://sqc/datasets/test.pkl datasets/test.pkl
gsutil -m cp gs://sqc/datasets/valid.pkl datasets/valid.pkl
python3 prepare_dataset.py
