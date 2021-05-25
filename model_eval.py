#################################################################################
#
#             Project Title:  Model Evaluator              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-05-24
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import pandas as pd
import numpy as np
import os
import itertools
import sys
from tqdm import tqdm
import torch

from tokenizer_model_factory import TokenizerModelFactory
from loader import Loader
from evaluator import CrisisEvaluator
#################################################################################
#   Function-Class Declaration
#################################################################################

def create_validation_labels(tokenizer, model, data):
    """Create evaluation predictions for a pretrained
    model and return the tensors for evaluation

    :model: Pretrained torch model
    :data: Dataset, likely a list of lists
    :returns: Set of predictions for the given data

    """
    device = torch.device('cuda')
    model = model.to(device)
    all_preds = []
    all_labels = []

    # Don't clutter GPU with gradients
    with torch.no_grad():

        #Iterate through dataset
        for i in tqdm(range(len(data))):
            epoch = data[i]
            batch = epoch[0]
            labels = epoch[1]

            #Get tokenization
            tokenized = tokenizer(list(batch), padding=True, is_split_into_words=True, return_length=True)
            input_ids = torch.tensor(tokenized["input_ids"]).to(device)
            attention_mask = torch.tensor(tokenized["attention_mask"]).to(device)
            labels_tensor = torch.tensor([list(label) +
                                        [-100]*(length - len(label)) for label, length
                                        in zip(labels, tokenized["length"])]).to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
            preds = outputs.logits.max(axis=-1)[1]
            mask = labels_tensor != -100
            preds = [p[l != -100].tolist() for p,l in zip(preds, labels_tensor)]
            # labels = [l[l != -100].tolist() for l in labels_tensor]

            all_preds = all_preds + preds
            # all_labels = all_labels + labels

    return all_preds


def read_in_loader_tok_model(path,
                  root ='/extra/datalab_scratch0/showrobl/models/',
                  ):
    """Read in model from filepath

    :path: Path to model
    :returns: loaded pretrained model

    """
    l = Loader()
    l.load_files()

    tkm = TokenizerModelFactory()
    tokenizer, model = tkm.makeMultilabelModel(path, root = root)
    return l, tokenizer, model


#################################################################################
#   Main
#################################################################################

if __name__ == "__main__":
    ROOT = '/extra/datalab_scratch0/showrobl/models/'
    loader, tokenizer, model = read_in_loader_tok_model('distilbert-base-uncased',
                                                        root = ROOT + 'multilabel/')
    ce = CrisisEvaluator(loader)
    data = loader.next_epoch(num_batches=100,
                                batch_size =32,
                                simulate = True,
                                dataset = "train")

    preds = create_validation_labels(tokenizer, model,data)
    ce.get_perf("Test", preds, data, kind="micro")
    print(ce.perf_dict)







