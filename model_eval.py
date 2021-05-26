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
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import sys
from tqdm import tqdm
import torch

from tokenizer_model_factory import TokenizerModelFactory
from loader import Loader
from evaluator import CrisisEvaluator
#################################################################################
#   Function-Class Declaration
#################################################################################

def create_validation_preds(tokenizer, model, data,
                            verbose = True):
    """Create evaluation predictions for a pretrained
    model and return the tensors for evaluation

    :model: Pretrained torch model
    :data: Dataset, likely a list of lists
    :returns: Set of predictions for the given data

    """
    device = torch.device('cuda')
    model = model.to(device).eval()
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
            # print(preds)
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

def bootstrap_multilabel_perf(path,
                              experiment_name,
                              ce = None,
                              trials = 10,
                              num_batches= 100,
                              batch_size =32,
                              dataset = "dev",
                              kinds = ["weighted","micro","macro"],
                              verbose = True,
                              root = '/extra/datalab_scratch0/showrobl/models/multilabel/'):
    """Boostrap performance for multilabel classification

    :path: Path to trained model
    :returns: CrisisEvaluator with Results

    """

    loader, tokenizer, model = read_in_loader_tok_model(path,
                                                        root = root)
    if not ce:
        ce = CrisisEvaluator(loader)

    for t in tqdm(range(trials)):
        data = loader.next_epoch(num_batches=num_batches,
                                    batch_size =batch_size,
                                    simulate = True,
                                    dataset = dataset,
                                    verbose = False)

        preds = create_validation_preds(tokenizer, model,data, verbose =verbose)
        for k in kinds:

            # Overall metrics
            ce.get_perf(experiment_name + "_{}".format(t),
                    preds,
                    data,
                    kind= k)

            # Per label metrics
            ce.get_per_label_perf(experiment_name + "_{}".format(t),
                    preds,
                    data,
                    kind= k)

            # Per crisis metrics
            ce.get_per_crisis_perf(experiment_name + "_{}".format(t),
                    preds,
                    data,
                    kind= k)

    return ce




#################################################################################
#   Main
#################################################################################

if __name__ == "__main__":
    ROOT = '/extra/datalab_scratch0/showrobl/models/multilabel/dis'

    trials = 100
    num_batches = 30
    paths = ['distilroberta-base','lstm','distilbert-base-uncased']
    datas = ['dev','test']
    for d in datas:
        for p in paths:
            print(p, d)
            ce = bootstrap_multilabel_perf(p,
                                   '{}_{}'.format(p,d),
                                   num_batches = num_batches,
                                   trials = trials,
                                   dataset = d)

            with open('artifacts/{}_all_perf_{}_t{}_b{}.pkl'
                      .format(p.replace("/","").split("-")[0],d,trials,num_batches),'wb') as file:
                pkl.dump(ce.perf_dict, file)








