# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model_dir = os.path.expanduser('/Users/xuming06/Documents/Data/chinese-xlnet-base/')

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelWithLMHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
sequence = f"少先队员应该"
input_ids = tokenizer.encode(sequence, return_tensors="pt")
# get logits of last hidden state
next_token_logits = model(input_ids).logits[:, -1, :]
# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
# sample
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
generated = torch.cat([input_ids, next_token], dim=-1)
resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
