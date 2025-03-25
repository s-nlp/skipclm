#!/bin/bash

# Define the directory containing the models
models_dir="gridsearch_models"

# Define the evaluation script path
eval_script="src/eval_flores.py"

# Define the number of parallel jobs
parallel_jobs=1

# Define the output results directory
results_dir="gridsearch_results"

# Dynamically generate the array of model directory paths
readarray -t model_dir_paths < <(find "${models_dir}" -maxdepth 1 -type d -name "gridsearch_xglm_564M_skip_*_*_tr")

# Check if any model directories were found
if [[ ${#model_dir_paths[@]} -eq 0 ]]; then
  echo "No model directories found in ${models_dir} matching the pattern 'gridsearch_xglm_564M_skip_*_*_tr'."
  exit 1
fi

# Create the results directory if it doesn't exist
mkdir -p "${results_dir}"

# Generate and pipe commands directly to parallel
for model_dir_path in "${model_dir_paths[@]}"; do
  # Extract just the directory name from the full path for output filename
  model_dir_name=$(basename "${model_dir_path}")
  model_path="${model_dir_path}" # Use the full path from find directly
  output_file="${results_dir}/output_${model_dir_name%/}.json"
  echo "python ${eval_script} --model_name ${model_path} --output_file ${output_file} --lang $1"
done | parallel -j ${parallel_jobs}

echo "All evaluation commands queued. Running in parallel with ${parallel_jobs} jobs."
echo "Check output files in: ${results_dir}/"
