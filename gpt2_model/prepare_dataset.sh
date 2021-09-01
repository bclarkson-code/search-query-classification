gsutil -m cp gs://search-query-classification-us-central1-c/final_raw/final_raw_train.pkl raw_datasets/final_raw_train.pkl
gsutil -m cp gs://search-query-classification-us-central1-c/final_raw/final_raw_test.pkl raw_datasets/final_raw_test.pkl
gsutil -m cp gs://search-query-classification-us-central1-c/final_raw/final_raw_valid.pkl raw_datasets/final_raw_valid.pkl
python3 prepare_dataset.py