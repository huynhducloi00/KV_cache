# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import random
import torch
from datasets import load_dataset
from tqdm import trange

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_wikitext2(nsamples, seed, seqlen, tokenizer, batch_size=1):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets

    traindata = traindata[:1000]

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        inps = torch.zeros((batch_size, seqlen)).long()
        tars = torch.zeros_like(inps)
        for b in range(batch_size):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            inps[b] = inp[0]
            tars[b] = tar[0]

        trainloader.append((inps, tars))
    return trainloader


def get_c4(nsamples, seed, seqlen, tokenizer, batch_size=1, from_cache=False):
    if from_cache:
        return torch.load('c4_train.pt', weights_only=True)
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    random.seed(seed)
    trainloader = []
    for _ in trange(nsamples):
        inps = torch.zeros((batch_size, seqlen)).long()
        tars = torch.zeros_like(inps)
        for b in range(batch_size):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            inps[b] = inp[0]
            tars[b] = tar[0]
        trainloader.append((inps, tars))
    return trainloader
