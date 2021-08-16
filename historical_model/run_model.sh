export XRT_TPU_CONFIG="localservice;0;localhost:51011"
unset LD_PRELOAD
if [ ! -d "datasets" ]; then
  gsutil cp gs://search-query-classification-us-central1-c/aol_input_data.zip .
  unzip -o aol_input_data.zip
  rm -f aol_input_data.zip
fi
python3 run_model.py
zip -r historical-logs.zip historical-model-logs
gsutil cp historical-logs.zip gs://search-query-classification-us-central1-c
gsutil cp lr_finder_results.pkl gs://search-query-classification-us-central1-c

