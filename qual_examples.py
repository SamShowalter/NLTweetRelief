#################################################################################
#
#             Project Title:  Qualitative Examples, Tweets              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-05-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import pickle as pkl
import numpy as np
from glob import glob
import pandas as pd
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer
from model_eval import *
#################################################################################
#   Function-Class Declaration
#################################################################################

def qual_example(filepath,
                 tokenizer, model,
                 path = None,
                 toks = None):
    """Load examples from filepath

    :filepath: Path to example file
    :returns: TODO

    """

    data = None
    with open(filepath, 'r') as file:
        data = file.read()

    data = [d for d  in data.split("\n") if len(d) > 0]
    tokens = [toks(d) for d in data]
    tokens = [t for t in tokens if len(t) >0]
    pseudo_labels = [[1]*len(t) for t in tokens]

    model_input = [tokens,pseudo_labels]

    preds = create_validation_preds(tokenizer,
                                    model,
                                    [model_input] ,
                                    demo = False,
                                    verbose = True,
                                    path = path)
    return tokens, preds









#################################################################################
#   Main Method
#################################################################################

if __name__ == "__main__":

    # # loader, tokenizer, model = read_in_loader_tok_model('distilbert-base-uncased')
    path ='distilbert-base-uncased'
    loader, tokenizer, model = read_in_loader_tok_model(path)

    # sample ="Our hearts go out to those affected by the fire that has injured 12 citizens. Several people are still missing and we will begin a search."
    toks = CountVectorizer().build_tokenizer()
    # sample = toks(sample)
    # preds = create_validation_preds(tokenizer, model,
    #                                 [[sample,
    #                                   [np.random.randint(0,10,size = len(sample))]]],
    #                                 verbose =True,path ='distilbert-base-uncased',demo = True )
    # print(sample)
    # print(preds)
    # print(loader.label_le.inverse_transform(preds[0]))
    files = glob('data/samples/*.txt')
    res_dict = {}
    for f in files:
        tag = f.split("/")[-1].split(".")[0]
        print(tag)
        res_dict[tag] = {}
        token, preds = qual_example(files[0], tokenizer, model, path = path,toks = toks)
        res_dict[tag]['tokens'] = token
        res_dict[tag]['preds'] = preds

    res_dict['le'] = loader.label_le
    with open('artifacts/qual_samples.pkl','wb') as file:
        pkl.dump(res_dict, file)










