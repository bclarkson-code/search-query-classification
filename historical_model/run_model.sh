export XRT_TPU_CONFIG="localservice;0;localhost:51011"
unset LD_PRELOAD
gsutil cp gs://search-query-classification-us-central1-c/aol_input_data.zip .
unzip -o aol_input_data.zip
rm -f aol_input_data.zip
python3 run_model.py