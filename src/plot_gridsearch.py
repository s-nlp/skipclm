import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Perform concurrent training runs over hyperparameters')
parser.add_argument('--lang', type=str, required=True, help='Language')

args = parser.parse_args()


# Directory containing your files
directory = './gridsearch_results'

# Metrics to consider
metrics_list = ['BLEU', 'COMET', 'METEOR', 'chrF', 'BERTScore', 'TER']

# Dictionary to store all metrics for each int_number_a
all_metrics = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith('output_gridsearch_') and filename.endswith(f'{args.lang}.json'):
        # Extract int_number_a and int_number_b from the filename
        parts = filename.split('_')
        int_number_a = int(parts[5])
        int_number_b = int(parts[6].split('.')[0])

        # Read JSON content from the file
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            file_metrics = {}
            for metric in metrics_list:
                if metric == 'BERTScore':
                    file_metrics[metric] = data.get(metric).get('F1')
                elif metric == 'TER':
                    file_metrics[metric] = 1 / data.get(metric)
                else:
                    file_metrics[metric] = data.get(metric)

        # Store the metrics in the dictionary
        if int_number_a not in all_metrics:
            all_metrics[int_number_a] = {'int_number_b': [], 'raw_metrics': {metric: [] for metric in metrics_list}}
        all_metrics[int_number_a]['int_number_b'].append(int_number_b)
        for metric in metrics_list:
            all_metrics[int_number_a]['raw_metrics'][metric].append(file_metrics[metric])

# Function to normalize a list of metric values to a 0-1 range
def normalize_metric(metric_values):
    valid_values = [v for v in metric_values if v is not None] # Filter out None values for normalization
    if not valid_values:
        return [None] * len(metric_values) # If no valid values, return list of None
    min_val = min(valid_values)
    max_val = max(valid_values)
    if max_val == min_val: # avoid division by zero if all values are the same
        return [0.5 if v is not None else None for v in metric_values] # Return middle value as all are same anyway.
    return [(v - min_val) / (max_val - min_val) if v is not None else None for v in metric_values]

# Normalize each metric for each int_number_a and calculate overall metric
filtered_metrics = {}
for int_number_a, data in all_metrics.items():
    if len(data['int_number_b']) >= 3 or True: # Keep all entries for now, original kept >= 3 or True
        filtered_metrics[int_number_a] = {'int_number_b': data['int_number_b']}
        normalized_metrics = {}
        for metric in metrics_list:
            normalized_metrics[metric] = normalize_metric(data['raw_metrics'][metric])
            filtered_metrics[int_number_a][metric] = normalized_metrics[metric] # Store normalized metrics

        # Calculate overall metric as the average of available normalized metrics
        overall_metric = []
        for i in range(len(data['int_number_b'])):
            valid_metric_sum = 0
            valid_metric_count = 0
            for metric in metrics_list:
                if normalized_metrics[metric][i] is not None:
                    valid_metric_sum += normalized_metrics[metric][i]
                    valid_metric_count += 1
            if valid_metric_count > 0:
                overall_metric.append(valid_metric_sum / valid_metric_count)
            else:
                overall_metric.append(None) # Handle case where no metric is available for a point
        filtered_metrics[int_number_a]['overall_metric'] = overall_metric


# Sort the data for each int_number_a and filter out None values for plotting
for int_number_a, data in filtered_metrics.items():
    # Combine int_number_b and overall_metric, filtering out None values
    combined_data = sorted([(b, o) for b, o in zip(data['int_number_b'], data['overall_metric']) if o is not None])
    filtered_metrics[int_number_a]['int_number_b'] = [item[0] for item in combined_data]
    filtered_metrics[int_number_a]['overall_metric'] = [item[1] for item in combined_data]


# Sort the keys (int_number_a) to ensure legend is ordered correctly
sorted_int_number_as = sorted(filtered_metrics.keys())

# Plot all data on a single graph
plt.figure(figsize=(12, 8))

for int_number_a in sorted_int_number_as:
    data = filtered_metrics[int_number_a]
    plt.plot(data['int_number_b'], data['overall_metric'], marker='o', label=r'$\alpha$ = ' + str(int_number_a), linewidth=3)

# Add labels and title with larger font size
plt.xlabel(r'$\beta$ value', fontsize=25, fontweight='bold')  # Use LaTeX for beta symbol
plt.ylabel('Overall Metric (Normalized)', fontsize=25, fontweight='bold')

# Determine the range of int_number_b and overall metrics
all_int_number_bs = [data['int_number_b'] for data in filtered_metrics.values() if data['int_number_b']] # Filter empty lists
all_overall_metrics = [data['overall_metric'] for data in filtered_metrics.values() if data['overall_metric']] # Filter empty lists


if all_int_number_bs and all_overall_metrics: # Check if there is data to plot
    min_beta = min(min(lst) for lst in all_int_number_bs if lst) if all_int_number_bs else 0
    max_beta = max(max(lst) for lst in all_int_number_bs if lst) if all_int_number_bs else 1 # Default to 1 if no beta values

    min_overall = min(min(lst) for lst in all_overall_metrics if lst) if all_overall_metrics else 0
    max_overall = max(max(lst) for lst in all_overall_metrics if lst) if all_overall_metrics else 1 # Default to 1 if no overall values

    # Set x-ticks and y-ticks with larger font size
    plt.xticks(range(min_beta, max_beta + 1, max(1, (max_beta - min_beta) // 5)), fontsize=25) # Ensure step is at least 1
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=25, va='center', rotation=90) # Fixed y-ticks for normalized metric

    # Sort legend by int_number_a values
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_labels, sorted_handles = zip(*sorted(zip(labels, handles), key=lambda x: int(x[0].split('= ')[1])))

    # Add the sorted legend to the plot with larger font size
    plt.legend(sorted_handles, sorted_labels, fontsize=16)

    # Add grid for better readability
    plt.grid(True)

    # Save the plot as a PDF
    plt.savefig('gridsearch_overall_metric.pdf', format='pdf')
    # plt.show()
    plt.close() # Close plot after saving to free memory
else:
    print("No valid metric data found to plot.")
