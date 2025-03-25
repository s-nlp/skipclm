#!/bin/bash

files=("./csvs/skip_flores.csv" "./csvs/noskip_flores.csv" "./csvs/nocontrastive_flores.csv" "./csvs/base_flores.csv")

for file in "${files[@]}"; do
  output_file="${file%.*}.xcomet-xl.csv"  # Create output name based on input
  echo "Processing $file..."
  python src/eval_comet.py "$file" -o "$output_file" -m "Unbabel/XCOMET-XL"
  echo "Finished processing $file. Results saved to $output_file"
  echo "----------------------------------------"
done

echo "All files processed."
