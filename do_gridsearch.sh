#!/bin/bash


# python gridsearch.py --min_param1 1 --max_param1 2 --min_param2 21 --max_param2 22 --lang $1
bash eval_gridsearch.sh $1
python src/plot_gridsearch.py --lang $1
