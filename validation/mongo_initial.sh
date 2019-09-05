#!/usr/bin/env bash
# Run mongo docker
docker run --name mongo -p 27017:27017 -v /Users/cchadha2/Documents/Github-Private/ieee_fraud/validation/fields.txt:/data/fields.txt -d mongo:latest