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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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
    tot_labels = []
    tot_preds = []
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
        raw_seq_weights = seq_weights
        seq_weights = np.array(seq_weights)/sum(seq_weights)
        all_labels = list(itertools.chain.from_iterable(labels))
        all_preds = list(itertools.chain.from_iterable(preds))
        tot_labels = tot_labels + all_labels
        tot_preds = tot_preds + all_preds
        perf_dict[c]['Precision'] = precision_score(all_labels, all_preds, average = 'micro')
        # print(perf_dict[c]['Precision'])
        perf_dict[c]['Recall'] = recall_score(all_labels, all_preds, average = 'micro')
        perf_dict[c]['F1'] = f1_score(all_labels, all_preds, average ='micro')

        perf_dict[c]['>1 Tag Seq. %'] = (np.array([crisis_dict[c][k]['1p_label_perc'] for k in keys])*seq_weights).sum()
        perf_dict[c]['Part. Seq. Acc.'] = (np.array([crisis_dict[c][k]['partially_correct_seq_perc'] for k in keys])*seq_weights).sum()
        perf_dict[c]['Tot. Seq. Acc.'] = (np.array([crisis_dict[c][k]['totally_correct_seq_perc'] for k in keys])*seq_weights).sum()
        perf_dict[c]['Tag Flips'] = (np.array([crisis_dict[c][k]['flip_count'] for k in keys])*seq_weights).sum()
        perf_dict[c]['Cont. Seq. Len.'] = (np.array([crisis_dict[c][k]['cont_seq_mean'] for k in keys])*seq_weights).sum()
        perf_dict[c]['Seq. Len.'] = len(all_preds)/sum(raw_seq_weights)
        # print(perf_dict[c]['mean_cont_seq_len'])

        # print(total_preds)
        # print(cm)
        # print(cm)
        perf_dict[c]['# Tokens'] = len(all_preds)

    raw_cm = confusion_matrix(tot_labels, tot_preds)
    total_preds = raw_cm.sum(axis = 1)
    cm = np.round((raw_cm.T/total_preds).T*100,1)
    cmdf = pd.DataFrame(cm)

    with open('artifacts/articles_conf_mat.tex', 'w') as tf:
        tf.write(cmdf.to_latex())

    df = pd.DataFrame(perf_dict).T
    df = df[['Precision', 'Recall','F1',  'Part. Seq. Acc.','Tot. Seq. Acc.','Tag Flips','>1 Tag Seq. %','Cont. Seq. Len.','Seq. Len.','# Tokens']]

    print(df.columns)
    np.set_printoptions(precision =2,suppress = True)
    print(np.round(df.to_numpy(), 2))


    with open('artifacts/article_perf_per_crisis.tex', 'w') as tf:
        tf.write(df.to_latex())




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
        # print([(len(p),len(l)) for p,l in zip(preds,labels) if len(p) != len(l)])
        partially_correct = [(p == l).any().astype(int) for p,l in zip(preds, labels)]
        totally_correct = [(p == l).all().astype(int) for p,l in zip(preds, labels)]

        perf_dict[k]['1p_label_perc'] = (uniques > 1).sum()/len(preds)
        perf_dict[k]['partially_correct_seq_perc'] = sum(partially_correct)/len(preds)
        perf_dict[k]['totally_correct_seq_perc'] = sum(totally_correct)/len(preds)
        # print(perf_dict[k]['1p_label_perc'])
        # print(perf_dict[k]['totally_correct_seq_perc'])
        # print(perf_dict[k]['partially_correct_seq_perc'])
        cont_seq_nums = continuous_sequence_nums(labels)
        flip_cnts = flip_count(labels)
        perf_dict[k]['flip_count'] = np.array(flip_cnts).mean()
        perf_dict[k]['cont_seq_mean'] = np.array(cont_seq_nums).mean()
        # print(perf_dict[k]['flip_count'])
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


def flip_count(seqs):
    all_flips = []
    for s in seqs:
        flips = 0
        seed_i = 0
        seed = s[0]
        for i in range(1,len(s)):
            if s[i] != seed:
                seed = s[i]
                flips += 1
                seed_i = i
        all_flips.append(flips)

    return all_flips




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

