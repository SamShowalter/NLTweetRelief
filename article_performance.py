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

    perf_dict = {}
    for c in crisis_dict.keys():
        print(c)
        perf_dict[c] = {}
        seq_weights = []
        labels = []
        preds = []
        for crisis in crisis_dict[c].keys():
            seq_weights.append(crisis_dict[c][crisis]['num_sequences'])
            preds.append(crisis_dict[c][crisis]['all_preds'])
            labels.append(crisis_dict[c][crisis]['all_labels'])


        keys = crisis_dict[c].keys()
        seq_weights = np.array(seq_weights)/sum(seq_weights)
        perf_dict[c]['1p_label_perc'] = (np.array([crisis_dict[c][k]['1p_label_perc'] for k in keys])*seq_weights).sum()
        perf_dict[c]['partially_correct_seq_perc'] = (np.array([crisis_dict[c][k]['partially_correct_seq_perc'] for k in keys])*seq_weights).sum()
        perf_dict[c]['totally_correct_seq_perc'] = (np.array([crisis_dict[c][k]['totally_correct_seq_perc'] for k in keys])*seq_weights).sum()
        perf_dict[c]['mean_cont_seq_len'] = (np.array([crisis_dict[c][k]['mean_cont_seq_len'] for k in keys])*seq_weights).sum()
        print(perf_dict[c]['1p_label_perc'])
        print(perf_dict[c]['partially_correct_seq_perc'])
        print(perf_dict[c]['totally_correct_seq_perc'])
        # print(perf_dict[c]['mean_cont_seq_len'])

        all_labels = list(itertools.chain.from_iterable(labels))
        all_preds = list(itertools.chain.from_iterable(preds))
        perf_dict[c]['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        cm = perf_dict[c]['confusion_matrix']
        total_preds = cm.sum(axis = 1)
        # print(total_preds)
        # print(cm)
        perf_dict[c]['confusion_matrix_perc_pred'] = np.round(cm.T/total_preds).T*100,2)
        perf_dict[c]['support'] = len(all_preds)





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
        # print(k)
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
        # print((np.array(perf_dict[k]['all_labels']) == 4).sum())

        # print(np.array(cont_seq_nums).mean())

    crisis_dict = {"hurricane_dorian":{}}

    for p in perf_dict.keys():
        if "can_wfire" not in p:
            # print(p)
            crisis_dict['hurricane_dorian'][p] = perf_dict[p]

    return crisis_dict




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
    crisis_dict =  get_article_performance(samples)
    combine_perf_metrics_by_crisis(crisis_dict)


    # print(samples.keys())
