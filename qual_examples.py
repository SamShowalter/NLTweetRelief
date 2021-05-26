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
import pandas as pd
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer
from model_eval import *
#################################################################################
#   Function-Class Declaration
#################################################################################

def qual_example(filepath):
    """Load examples from filepath

    :filepath: Path to example file
    :returns: TODO

    """
    pass


#################################################################################
#   Main Method
#################################################################################

if __name__ == "__main__":

    # loader, tokenizer, model = read_in_loader_tok_model('distilbert-base-uncased')
    loader, tokenizer, model = read_in_loader_tok_model('distilbert-base-uncased')
    sample ="Our hearts go out to those affected by the fire that has injured 12 citizens. Several people are still missing and we will begin a search."
    toks = CountVectorizer().build_tokenizer()
    sample = toks(sample)
    preds = create_validation_preds(tokenizer, model,
                                    [[sample,
                                      [np.random.randint(0,10,size = len(sample))]]],
                                    verbose =True,path ='distilbert-base-uncased' )
    print(sample)
    print(preds)
    print(loader.label_le.inverse_transform(preds[0]))


