export XRT_TPU_CONFIG="localservice;0;localhost:51011"
unset LD_PRELOAD
mkdir datasets
gsutil -m cp gs://search-query-classification-us-central1-c/aol_data_* datasets
python3 prepare_dataset.py
