zip -r pretrain-logs.zip pretrain-logs
gsutil cp pretrain-logs.zip gs://search-query-classification-europe-west4-a
rm -f pretrain-logs.zip