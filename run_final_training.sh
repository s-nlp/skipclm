#!/bin/bash


# python src/train_transformers.py --type skip --skip_start 3 --skip_end 18 --lang tr --gridsearch_addition "final_models/"
# python src/train_transformers.py --type noskip --skip_start 1 --skip_end 19 --lang de --gridsearch_addition "final_models/" --reverse_direction
# python src/train_transformers.py --type nocontrastive --skip_start 1 --skip_end 19 --lang de --gridsearch_addition "final_models/" --reverse_direction
# python src/train_transformers.py --type skip --skip_start 1 --skip_end 20 --lang zh --gridsearch_addition "final_models/"
# python src/train_transformers.py --type noskip --skip_start 1 --skip_end 19 --lang zh --gridsearch_addition "final_models/" --reverse_direction
# python src/train_transformers.py --type nocontrastive --skip_start 1 --skip_end 19 --lang zh --gridsearch_addition "final_models/" --reverse_direction
# python src/train_transformers.py --type skip --skip_start 1 --skip_end 20 --lang tr --gridsearch_addition "final_models/"
# python src/train_transformers.py --type noskip --skip_start 1 --skip_end 19 --lang tr --gridsearch_addition "final_models/" --reverse_direction
# python src/train_transformers.py --type nocontrastive --skip_start 1 --skip_end 19 --lang tr --gridsearch_addition "final_models/" --reverse_direction

python src/eval_flores.py --model_name final_models/xglm_564M_skip_3_18_tr --output_file final_models_evals/xglm_564M_skip_3_18_tr.json --lang tr

# python src/eval_flores.py --model_name final_models/xglm_564M_skip_1_19_de --output_file final_models_evals/xglm_564M_skip_1_19_de_en.json --lang de & python src/eval_flores.py --model_name final_models/xglm_564M_skip_1_19_zh --output_file final_models_evals/xglm_564M_skip_1_19_zh_en.json --lang zh
# python src/eval_flores.py --model_name final_models/xglm_564M_skip_1_19_tr --output_file final_models_evals/xglm_564M_skip_1_19_tr_en.json --lang tr & python src/eval_flores.py --model_name final_models/xglm_564M_noskip_1_19_de --output_file final_models_evals/xglm_564M_noskip_1_19_de_en.json --lang de 
# python src/eval_flores.py --model_name final_models/xglm_564M_noskip_1_19_zh --output_file final_models_evals/xglm_564M_noskip_1_19_zh_en.json --lang zh & python src/eval_flores.py --model_name final_models/xglm_564M_noskip_1_19_tr --output_file final_models_evals/xglm_564M_noskip_1_19_tr_en.json --lang tr
# python src/eval_flores.py --model_name final_models/xglm_564M_nocontrastive_1_19_de --output_file final_models_evals/xglm_564M_nocontrastive_1_19_de_en.json --lang de & python src/eval_flores.py --model_name final_models/xglm_564M_nocontrastive_1_19_zh --output_file final_models_evals/xglm_564M_nocontrastive_1_19_zh_en.json --lang zh 
# python src/eval_flores.py --model_name final_models/xglm_564M_nocontrastive_1_19_tr --output_file final_models_evals/xglm_564M_nocontrastive_1_19_tr_en.json --lang tr
