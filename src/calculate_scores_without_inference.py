from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import argparse
import datasets
import eval_comet
import json
import nltk
import os
import pandas as pd
import pathlib
import sacrebleu
import torch
import tqdm


def prepare_dataset(parallel_data, tokenizer):
    return datasets.Dataset.from_list([
        {'source': item['sent0'], 'target': item['sent1']}
        for item in parallel_data
        if tokenizer(item['sent0'], return_tensors='pt', truncation=True, max_length=128)['input_ids'].squeeze().shape[0] + tokenizer(item['sent1'], return_tensors='pt', truncation=True, max_length=128)['input_ids'].squeeze().shape[0] <= 256
    ])

# Load the test data
def load_sentence_pairs(tokenizer, lang):
    # Load the parallel data from JSON file and
    # use only the top 1000 sentence pairs
    lang_code_to_name = {
        'ru': 'Russian',
        'de': 'German',
        'zh': 'Chinese',
        'tr': 'Turkish',
    }

    lang_code_to_flores = {
        'ru': 'rus',
        'de': 'deu',
        'zh': 'zho_trad',
        'tr': 'tur',
    }

    eng_file = "flores101_dataset/devtest/eng.devtest"
    rus_file = f"flores101_dataset/devtest/{lang_code_to_flores[lang]}.devtest"
    with open(eng_file, 'r', encoding='utf-8') as f1, open(rus_file, 'r', encoding='utf-8') as f2:
        eng_lines = f1.readlines()
        rus_lines = f2.readlines()
    parallel_data = [{'sent0': eng, 'sent1': rus} for eng, rus in zip(eng_lines, rus_lines)]
    dataset = prepare_dataset(parallel_data, tokenizer)
    sentence_pairs = [('English: ' + item['source'], f'{lang_code_to_name[lang]}: ' + item['target']) for item in dataset]

    return sentence_pairs

# Load the model and tokenizer
def load_model_and_tokenizer(model_name):
    if 'noskip' not in model_name and not 'nocontrastive' in model_name and model_name != 'facebook/xglm-564M':
        skip_start, skip_end = model_name.split('_')[-3:-1]
        skip_start, skip_end = int(skip_start), int(skip_end)
        model = XGLMWithSkipConnectionForCausalLM.from_pretrained(model_name, config=XGLMWithSkipConnectionConfig(skip_start=skip_start, skip_end=skip_end)).to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda')

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('facebook/xglm-564M')

    return model, tokenizer

# Generate translations
def generate_translations(model, tokenizer, sources):
    translations = []
    for source in tqdm.tqdm(sources):
        inputs = tokenizer(source, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, eos_token_id=tokenizer.pad_token_id, temperature=0.0)
        translation = tokenizer.decode(output_ids[0][len(inputs[0]):], skip_special_tokens=True)

        translations.append(translation)
    return translations

# Calculate BLEU score
def calculate_bleu_score(generated_translations, references, lang):
    if lang == 'zh':
        bleu = sacrebleu.corpus_bleu(generated_translations, [references], tokenize='zh')
    else:
        bleu = sacrebleu.corpus_bleu(generated_translations, [references])
    return bleu.score

# Calculate METEOR score
def calculate_meteor_score(generated_translations, references):
    if not os.path.exists('nltk_wordnet_downloaded'):
        nltk.download('wordnet')
        with open('nltk_wordnet_downloaded', 'w') as f:
            f.write('')
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(hyp)) for ref, hyp in zip(references, generated_translations)]
    return sum(meteor_scores) / len(meteor_scores)

# Calculate chrF score
def calculate_chrf_score(generated_translations, references):
    chrf = sacrebleu.corpus_chrf(generated_translations, [references])
    return chrf.score

# Calculate BERTScore
def calculate_bert_score(generated_translations, references):
    P, R, F1 = bert_score(tuple(generated_translations), tuple(references), lang="all")
    return P.mean(), R.mean(), F1.mean()

# Calculate TER score
def calculate_ter_score(generated_translations, references):
    from sacrebleu.metrics import TER
    ter_metric = TER()
    ter = ter_metric.corpus_score(generated_translations, [references])
    return ter.score

# Main function
def calculate_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_predictions", type=str, required=True, help="Path to model")
    parser.add_argument("--output_file", type=str, default="scores.json", help="File to save the evaluation scores.")
    parser.add_argument("--lang", type=str, required=True, help="Language to use for evaluation.")

    args = parser.parse_args()

    # Load data
    sources, generated_translations, references = list(zip(*map(lambda x: x.split('\t'), pathlib.Path(args.path_to_predictions).read_text().splitlines())))

    # Calculate scores
    bleu_score = calculate_bleu_score(generated_translations, references, args.lang)
    meteor_score = calculate_meteor_score(generated_translations, references)
    chrf_score = calculate_chrf_score(generated_translations, references)
    P, R, F1 = calculate_bert_score(generated_translations, references)

    ter_score = calculate_ter_score(generated_translations, references)

    # Save scores to file
    results = {
        "BLEU": bleu_score,
        "METEOR": meteor_score,
        "chrF": chrf_score,
        "BERTScore": {"Precision": P.item(), "Recall": R.item(), "F1": F1.item()},
        "TER": ter_score
    }
    print(json.dumps(results, indent=2))
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    df, comet_score = eval_comet.calculate_comet_scores(args.output_file.split('.')[0] + '.csv', 'Unbabel/wmt22-comet-da')
    results['COMET'] = comet_score

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    df.to_csv(args.output_file.split('.')[0] + '_comet' + '.csv', index=False)

    print(args.output_file)
    print(json.dumps(results, indent=2))

    with open(args.output_file.split('.')[0] + '.csv', 'w') as f:
        f.write('\n'.join([f'{src if not src.endswith("\n") else src[:-1]}\t{ref}\t{tr}' for src, ref, tr in zip(sources, generated_translations, references)]))

if __name__ == "__main__":
    calculate_scores()
