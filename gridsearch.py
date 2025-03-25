import subprocess
import argparse
import itertools
import tqdm
import multiprocessing

def run_script_concurrent(hyperparam1, hyperparam2, lang):
    """
    Run the bash script with the given hyperparameters, without parsing output.
    """
    try:
        print(f"Starting training for hyperparams: ({hyperparam1}, {hyperparam2}, {lang})")
        subprocess.check_call(f"./run_training.sh {hyperparam1} {hyperparam2} {lang}", shell=True)
        print(f"Finished training for hyperparams: ({hyperparam1}, {hyperparam2}, {lang})")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred for params ({hyperparam1}, {hyperparam2}, {lang}): {e}")
        return False
    return True


def concurrent_training(min_param1, max_param1, min_param2, max_param2, lang):
    """
    Run trainings concurrently for all hyperparameter combinations.
    """
    param_combinations = [_ + (lang,) for _ in itertools.product(range(min_param1, max_param1 + 1), range(min_param2, max_param2 + 1))]

    with multiprocessing.Pool(processes=1) as pool: # Use a pool of 2 processes for concurrent training
        tasks = []
        for params in param_combinations:
            tasks.append(params)

        with tqdm.tqdm(total=len(tasks)) as pbar:
            for _ in pool.starmap(run_script_concurrent, tasks): # Use starmap to pass params as separate arguments
                pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform concurrent training runs over hyperparameters')
    parser.add_argument('--min_param1', type=int, required=True, help='Minimum value of hyperparameter 1')
    parser.add_argument('--max_param1', type=int, required=True, help='Maximum value of hyperparameter 1')
    parser.add_argument('--min_param2', type=int, required=True, help='Minimum value of hyperparameter 2')
    parser.add_argument('--max_param2', type=int, required=True, help='Maximum value of hyperparameter 2')
    parser.add_argument('--lang', type=str, required=True, help='Language')
    args = parser.parse_args()

    concurrent_training(args.min_param1, args.max_param1, args.min_param2, args.max_param2, args.lang)
