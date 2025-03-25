from transformers import XGLMTokenizer
import transformers
from accelerate.utils import set_seed
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import random
from configuration import XGLMWithSkipConnectionConfig
from decoder import XGLMWithSkipConnectionForCausalLM

set_seed(42)

def set_seed_stuff(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Set a fixed seed
set_seed_stuff(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# If you're using operations that are non-deterministic on CUDA, 
# you can use the following (note: this may impact performance)
torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    from train_transformers import ParallelDataset, custom_collate_fn
    import json
    import pathlib
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data import DataLoader

    model = transformers.XGLMForCausalLM.from_pretrained('facebook/xglm-564M')
    skip_model = XGLMWithSkipConnectionForCausalLM.from_pretrained('facebook/xglm-564M', config=XGLMWithSkipConnectionConfig())

    parallel_data = json.loads('[' + ','.join(pathlib.Path("AFP.en_zh.X0.5.R3.0.json").read_text().splitlines()) + ']')
    tokenizer = XGLMTokenizer.from_pretrained('facebook/xglm-564M')
    dataset = ParallelDataset(parallel_data, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: custom_collate_fn(x, tokenizer=tokenizer))

    for batch in loader:
        model.eval()
        skip_model.eval()
        hs = model.forward(**batch['anchor_ids'], output_hidden_states=True).hidden_states
        shs = skip_model.forward(**batch['anchor_ids'], output_hidden_states=True).hidden_states
        for idx, (sh, h) in enumerate(zip(shs, hs)):
            assert torch.allclose(sh, h), idx
        print('Forward in eval done')

        model.train()
        skip_model.train()
        assert torch.allclose(skip_model(**batch['anchor_ids']).last_hidden_state, model(**batch['anchor_ids']).last_hidden_state)
        print('Forward in traindone')
        break
