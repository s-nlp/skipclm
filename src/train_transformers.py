import argparse
import torch
import typing
import json
import pathlib
from transformers import XGLMTokenizer, XGLMForCausalLM, Trainer, TrainingArguments
import transformers
from decoder import XGLMWithSkipConnectionForCausalLM
from configuration import XGLMWithSkipConnectionConfig
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import datasets
from oml.losses.triplet import TripletLoss
import os 
import gc
import wandb

import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Get a custom dataset
def prepare_dataset(parallel_data, tokenizer, reverse_direction=False):
    if not reverse_direction:
        return datasets.Dataset.from_list([
            {'source': item['sent0'], 'target': item['sent1']}
            for item in parallel_data
            if tokenizer(item['sent0'], return_tensors='pt')['input_ids'].squeeze().shape[0] + tokenizer(item['sent1'], return_tensors='pt')['input_ids'].squeeze().shape[0] <= 512
        ])
    else:
        return datasets.Dataset.from_list([
            {'source': item['sent1'], 'target': item['sent0']}
            for item in parallel_data
            if tokenizer(item['sent0'], return_tensors='pt')['input_ids'].squeeze().shape[0] + tokenizer(item['sent1'], return_tensors='pt')['input_ids'].squeeze().shape[0] <= 512
        ])


def triplet_loss(anchor, positive, negatives, temperature=0.1):
    """
    Computes a contrastive triplet loss using cosine similarity and CrossEntropyLoss.

    Args:
        anchor (torch.Tensor): Anchor embeddings, shape (batch_size, embedding_dim)
        positive (torch.Tensor): Positive embeddings, shape (batch_size, embedding_dim)
        negatives (torch.Tensor): Negative embeddings, shape (batch_size, num_negatives, embedding_dim) or (batch_size, embedding_dim)
        temperature (float): Temperature scaling for softmax.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size = anchor.size(0)

    # Handle single negative by adding an extra dimension
    if negatives.dim() == 2:
        negatives = negatives.unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)

    # Normalize embeddings
    anchor_norm = F.normalize(anchor, p=2, dim=1)             # (batch_size, embedding_dim)
    positive_norm = F.normalize(positive, p=2, dim=1)         # (batch_size, embedding_dim)
    negatives_norm = F.normalize(negatives, p=2, dim=2)       # (batch_size, num_negatives, embedding_dim)

    # Compute cosine similarity for anchor-positive
    ap_sim = torch.sum(anchor_norm * positive_norm, dim=1)    # (batch_size,)

    # Compute cosine similarity for anchor-negatives
    an_sim = torch.sum(anchor_norm.unsqueeze(1) * negatives_norm, dim=2)  # (batch_size, num_negatives)

    # Concatenate positive similarities with negative similarities
    logits = torch.cat([ap_sim.unsqueeze(1), an_sim], dim=1)    # (batch_size, 1 + num_negatives)

    # Apply temperature scaling
    # logits /= temperature

    # Labels: positive is the first class (index 0)
    labels = torch.zeros(batch_size, dtype=torch.long).to(anchor.device)  # (batch_size,)

    # Compute CrossEntropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


class TripletLossTrainer(Trainer):
    def __init__(self, type_of_training: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_of_training = type_of_training

        self.triplet_loss = TripletLoss(margin=0.1)
    
    def compute_loss(self, model, inputs, **kwargs):
        def get_eos_embeddings(input_ids, hidden_states):
            # Batch size and sequence length
            batch_size, seq_len = input_ids.shape
            
            # Get the EOS token ID from the last token of each sentence (guaranteed to be EOS)
            eos_token_ids = input_ids[:, -1].unsqueeze(1)  # Shape: (32, 1)
            
            # Create a boolean mask where input_ids equals the respective EOS token
            eos_mask = (input_ids == eos_token_ids)  # Shape: (32, N)

            # Find the index of the first occurrence of EOS token in each sequence
            # torch.argmax will return the first occurrence of True (which is treated as 1)
            first_eos_indices = eos_mask.float().argmax(dim=1)  # Shape: (32,)

            # Gather the hidden states corresponding to the first EOS token for each sentence
            first_eos_hidden_states = hidden_states[torch.arange(batch_size), first_eos_indices]

            return first_eos_hidden_states

        anchor_ids = inputs['anchor_ids']
        positive_ids = inputs['positive_ids']
        negative_ids = inputs['negative_ids']

        outputs = model(**anchor_ids, labels=anchor_ids['input_ids'])
        lm_loss = outputs.loss

        outputs_anchor = model.forward_encoder(**anchor_ids).last_hidden_state
        outputs_positive = model.forward_encoder(**positive_ids).last_hidden_state
        outputs_negative = model.forward_encoder(**negative_ids).last_hidden_state

        anchor_embeddings = get_eos_embeddings(anchor_ids['input_ids'], outputs_anchor)
        positive_embeddings = get_eos_embeddings(anchor_ids['input_ids'], outputs_negative)
        negative_embeddings = get_eos_embeddings(anchor_ids['input_ids'], outputs_positive)

        triplet_loss_val = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        loss = lm_loss + 0.01 * triplet_loss_val if self.type_of_training != 'nocontrastive' else lm_loss

        del outputs
        del outputs_anchor
        del outputs_positive
        del outputs_negative
        del anchor_embeddings
        del positive_embeddings
        del negative_embeddings

        gc.collect()
        torch.cuda.empty_cache()

        return loss


class DataCollator:
    def __init__(self, tokenizer, training_type):
        self.tokenizer = tokenizer
        self.training_type = training_type
    
    def _get_not_this(self, item, container):
        smth = random.randint(0, len(container) - 1)
        while container[smth] == item:
            smth = random.randint(0, len(container) - 1)
        return container[smth]

    def __call__(self, batch: typing.List[dict]):
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]

        # Anchor: source + target
        anchors = [source + target + self.tokenizer.pad_token for source, target in zip(sources, targets)]
        # Positive: source + source
        positives = [source + ' ' + source + self.tokenizer.pad_token for source in sources]
        # Negative: source + non-source OR source + non-target
        negatives = [
            source + ' ' + random.choice(
                [
                    self._get_not_this(source, sources),
                    self._get_not_this(target, targets), 
                ]
            ) + tokenizer.pad_token 
            for source, target in zip(sources, targets)
        ]

        max_length = max(
            (self.tokenizer(_, padding=True, return_tensors='pt')['input_ids'].shape[1] for _ in (anchors, positives, negatives))
        )
        
        # Tokenizing
        anchor_ids = self.tokenizer(anchors, padding='max_length', max_length=max_length, return_tensors='pt')
        positive_ids = self.tokenizer(positives, padding='max_length', max_length=max_length, return_tensors='pt')
        negative_ids = self.tokenizer(negatives, padding='max_length', max_length=max_length, return_tensors='pt')

        del anchors
        del positives
        del negatives

        gc.collect()
        torch.cuda.empty_cache()
        if self.training_type != 'nocontrastive':
            return {'anchor_ids': anchor_ids , 'positive_ids': positive_ids, 'negative_ids': negative_ids}
        else:
            return {'input_ids': anchor_ids['input_ids'], 'attention_mask': anchor_ids['attention_mask'], 'labels': anchor_ids['input_ids']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['skip', 'noskip', 'nocontrastive'],
        required=True,
        help="Specify the type: 'skip', 'noskip', or 'nocontrastive'."
    )

    parser.add_argument(
        '--skip_start',
        type=int,
        required=True,
        help="Skip start layer/contrastive layer."
    )

    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help="Language code."
    )

    parser.add_argument(
        '--gridsearch_addition',
        type=str,
        default='',
        required=False,
        help="Language code."
    )

    parser.add_argument(
        '--skip_end',
        type=int,
        required=True,
        help="Skip end layer."
    )

    parser.add_argument('--reverse_direction', action='store_true')
    args = parser.parse_args()

# Load the dataset and create a data loader
    parallel_data = json.loads('[' + ','.join(pathlib.Path(f"AFP.en_{args.lang}.X0.5.R3.0.json").read_text().splitlines()) + ']')
    parallel_data = parallel_data[:len(parallel_data) // 2]

    tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-564M')
    data_collator = DataCollator(tokenizer, args.type)

    dataset = prepare_dataset(parallel_data, tokenizer, reverse_direction=args.reverse_direction)
# Initialize the model, optimizer, and device
    training_args = TrainingArguments(
        output_dir=f'./{args.gridsearch_addition}xglm_564M_{args.type}_{args.skip_start}_{args.skip_end}_{args.lang}',  # output directory
        num_train_epochs=1,  # number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        save_strategy='epoch',
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        logging_steps=100,
        # evaluation_strategy='steps',
        # eval_steps=1000,
        save_total_limit=3,
        # load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        save_on_each_node=True,
        remove_unused_columns=False,
        seed=42,
    )

    if args.type == 'skip':
        from decoder import XGLMWithSkipConnectionForCausalLM
        skip_model = XGLMWithSkipConnectionForCausalLM.from_pretrained('facebook/xglm-564M', config=XGLMWithSkipConnectionConfig(skip_start=args.skip_start, skip_end=args.skip_end, lambda_warmup_steps=300))
    elif args.type == 'noskip':
        from decoder_noskip import XGLMWithSkipConnectionForCausalLM
        skip_model = XGLMWithSkipConnectionForCausalLM.from_pretrained('facebook/xglm-564M', config=XGLMWithSkipConnectionConfig(skip_start=args.skip_start, skip_end=args.skip_end, lambda_warmup_steps=300))
    else:
        skip_model = transformers.AutoModelForCausalLM.from_pretrained('facebook/xglm-564M')
    
    run = wandb.init(
        name=f"xglm-564M-{args.type}",
        project="skipclm",
        tags=[args.type],
    )
    wandb.watch(skip_model, log=None, log_freq=10)
    
    if args.type in ['skip', 'noskip']:
        trainer = TripletLossTrainer(
            model=skip_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            type_of_training=args.type,
        )
    else:
        trainer = Trainer(
            model=skip_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )


    # Train the model
    trainer.train()
    run.finish()

    trainer.save_model()
