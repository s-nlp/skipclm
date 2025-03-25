import pandas as pd
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import argparse
import json
import numpy as np


def calculate_comet_scores(csv_filepath, model_path="Unbabel/wmt22-comet-da", reverse=False):
    """
    Calculates COMET scores for translations in a CSV file and returns the mean system score.

    Args:
        csv_filepath: Path to the CSV file.
        model_path: Path or name of the COMET model.

    Returns:
        tuple: A tuple containing the pandas DataFrame with COMET scores and the mean system score.
        Returns None, None if an error occurs.
    """

    try:
        df = pd.read_csv(csv_filepath, sep='\t', header=None, names=['src', 'mt', 'ref'])
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return None, None
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file at {csv_filepath}. Check the delimiter.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return None, None

    try:
        model_path = download_model(model_path)
        model = load_from_checkpoint(model_path)
    except Exception as e:
        print(f"Error loading COMET model: {e}")
        return None, None

    data = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating COMET scores"):
        try:
            if row['ref'] is not np.nan:
                data.append({
                    "src": row['src'],
                    "mt": row['mt'],
                    "ref": row['ref']
                })
            else:
                data.append({
                    "src": row['src'],
                    "mt": row['src'],
                    "ref": row['ref']
                })
        except TypeError:
            print(row)
            raise

    print(data)
    try:
        model_output = model.predict(data, batch_size=8, gpus=1)  # Adjust batch size and gpus as needed. gpus=0 for CPU
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error. Try reducing the batch size or using CPU (gpus=0).")
            return None, None
        else:
            print(f"A RuntimeError occurred during prediction: {e}")
            return None, None
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        print(json.dumps(data[66*8:68*8], indent=2, ensure_ascii=False))
        return None, None

    df['comet_score'] = [seg for seg in model_output.scores]
    system_score = model_output.system_score  # Extract the system-level score
    return df, system_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate COMET scores for translations in a CSV file.")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", help="Path to the output CSV file (optional).", default="comet_results.csv")
    parser.add_argument("-m", "--model", help="Path or name of the COMET model (optional).", default="Unbabel/wmt22-comet-da")
    args = parser.parse_args()

    csv_file = args.csv_file
    output_file = args.output
    model_path = args.model

    result_df, system_score = calculate_comet_scores(csv_file, model_path)

    if result_df is not None:
        print(result_df)
        result_df.to_csv(output_file, index=False)
        print(f"COMET scores saved to {output_file}")
        print(f"Overall COMET Score for the dataset: {system_score:.4f}") # Print the overall COMET score formatted to 4 decimal places
