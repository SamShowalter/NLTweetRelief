#################################################################################
#
#             Project Title:  Selected Article Performance              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-06-02
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import itertools
from sklearn.metrics import confusion_matrix

#################################################################################
#   Function-Class Declaration
#################################################################################

'''
Types of performance metrics to track:
    - What percentage of sequences have > 1 label (are not unilabel)
    - What percentage of sequences are completely wrong
    - Average number of flips in a sequence
    - Avg. length of continuous topic
    - Top 3 failure modes (by percentage)


    - OPTIONAL: Top k performance ('other_relevant_info' <-> 'not_humanitarian')

'''

def combine_perf_metrics_by_crisis(crisis_dict,
                                   tok_mets = [],
                                   seq_mets = []):

    seq_weights = []
    tok_weights = []
    for c in crisis_dict.keys():
        seq_weights.append(crisis_dict[c]['num_sequences'])
        tok_weights.append(crisis_dict[c]['num_tokens'])

    seq_weights = np.array(seq_weights)/sum(seq_weights)
    tok_weights = np.array(tok_weights)/sum(tok_weights)


def get_article_performance(res_dict):
    """Get article performance beyond precision and recall.
    Consider how often topics change mid-sentence, as well as how
    often errors occur across a whole sequence or at threshold

    :res_dict: Results dictionary of articles
    :mets: Metrics to test (some will be custom-made)
    :crises: list of held-out crises
    :returns: performance_dictionary

    """
    perf_dict = {}
    for k in res_dict.keys():
        preds  = [np.array(r) for r in res_dict[k][0]]
        labels = [np.array(r) for r in res_dict[k][1]]
        perf_dict[k] = {}
        perf_dict[k]['num_sequences'] = len(preds)
        perf_dict[k]['num_tokens'] = sum([a.sum() for a in preds])
        uniques = np.array([len(np.unique(a)) for a in labels])
        partially_correct = [(p == l).any().astype(int) for p,l in zip(preds, labels)]
        totally_correct = [(p == l).all().astype(int) for p,l in zip(preds, labels)]

        perf_dict[k]['1p_label_perc'] = (uniques > 1).sum()/len(preds)
        perf_dict[k]['partially_correct_seq_perc'] = sum(partially_correct)/len(preds)
        perf_dict[k]['totally_correct_seq_perc'] = sum(totally_correct)/len(preds)
        # print(perf_dict[k]['1p_label_perc'])
        # print(perf_dict[k]['totally_correct_seq_perc'])
        # print(perf_dict[k]['partially_correct_seq_perc'])
        cont_seq_nums = continuous_sequence_nums(labels)
        perf_dict[k]['mean_cont_seq_len'] = np.array(cont_seq_nums).mean()
        perf_dict[k]['all_labels'] = list(itertools.chain.from_iterable(labels))
        perf_dict[k]['all_preds'] = list(itertools.chain.from_iterable(preds))

        print(np.array(cont_seq_nums).mean())



def continuous_sequence_nums(seqs):
    all_seqs = []
    for s in seqs:
        seq_lens = []
        seed_i = 0
        seed = s[0]
        for i in range(1,len(s)):
            if s[i] != seed:
                seed = s[i]
                seq_lens.append(i-seed_i)
                seed_i = i
        seq_lens.append(len(s) - seed_i)
        all_seqs = all_seqs + seq_lens

    return all_seqs




def read_pkl(filepath,
             root = 'artifacts/'):
    with open(root + filepath, 'rb') as file:
        return pkl.load(file)


#################################################################################
#   Main Method
#################################################################################

if __name__ == "__main__":
    samples = read_pkl('sams_articles_preds_golds.pkl')
    get_article_performance(samples)


    # print(samples.keys())

