export XRT_TPU_CONFIG="localservice;0;localhost:51011"
unset LD_PRELOAD
if [ ! -d "datasets" ]; then
  mkdir datasets
  gsutil -m cp gs://search-query-classification-us-central1-c/aol_data_* datasets
fi
if [ ! -d "gpt2-checkpoints" ]; then
  gsutil -m cp gs://search-query-classification-us-central1-c/gpt2-checkpoints.zip .
  unzip gpt-checkpoints.zip
fi

python3 prepare_dataset.py
