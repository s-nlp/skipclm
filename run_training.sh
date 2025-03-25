#!/bin/bash

python src/train_transformers.py --type skip --skip_start $1 --skip_end $2 --lang $3 --gridsearch_addition "gridsearch_models/gridsearch_"

