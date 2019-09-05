#!/usr/bin/env bash

# Dump mongo archive
docker exec mongo sh -c 'exec mongodump -d validation --archive' > /Users/cchadha2/Documents/Github-Private/ieee_fraud/validation/data/lgb_valid.archive

# Dump archive as csv
docker exec mongo sh -c 'mongoexport --db validation --collection feature_engineering --type csv --fieldFile /data/fields.txt' > /Users/cchadha2/Documents/Github-Private/ieee_fraud/validation/data/lgb_valid.csv