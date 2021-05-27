#################################################################################
#
#             Project Title:  Generate Latex Tables              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-05-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import pandas as pd
import numpy as np
import tabulate
from glob import glob
import pickle as pkl
import itertools

#################################################################################
#   Function-Class Declaration
#################################################################################

def make_perf_table(filepaths):
    """Make general performance table

    :filepath: Path to file

    """
    started = False
    filepaths.sort()
    for i in range(0, len(filepaths), 2):
        path_dev = filepaths[i]
        path_test = filepaths[i+1]
        data_row = []
        for path,data_type in zip([path_dev,path_test],
                        ['dev','test']):
            data = read_pkl(path)
            collected_metrics = pd.DataFrame(
                [data[k]['weighted'] for k in data.keys()])
            collected_metrics['support'] = collected_metrics['confusion_matrix'].apply(lambda x: x.sum())
            collected_metrics.drop('confusion_matrix',inplace = True,axis =1)
            mean_info = collected_metrics.mean(axis = 0).astype(float)
            mean_info['type'] = 'mean'
            std_info = collected_metrics.std(axis = 0).astype(float)
            std_info['type'] = 'std'
            row_data = pd.concat([mean_info, std_info],
                            axis = 1).T
            row_data = row_data.add_prefix(data_type +"_")
            if 'type' not in row_data.columns:
                row_data['type'] = row_data['{}_type'.format(data_type)]
            row_data.drop(['{}_type'.format(data_type)],inplace = True, axis = 1)
            row_data['model'] = path.split("/")[-1].split("_")[0]
            data_row.append(row_data)

        if not started:
            started = True
            df = pd.concat(data_row,
                            axis = 1)
        else:
            new_row = pd.concat(data_row,
                            axis = 1)
            df = pd.concat([df,new_row],axis = 0)

    # Now let's make latex table
    mets = ['precision','recall','accuracy','F1','support']

    df.reset_index(inplace = True, drop = True)
    tdf = df
    t = tdf['type']
    model = tdf['model'].iloc[:,0]
    tdf.drop(['type','model'],inplace = True, axis = 1)
    tdf = tdf.astype(float)
    # tdf = tdf.round(2)
    # tdf[:,tdf.columns != 'type'] =tdf[:,tdf.columns != 'type'].astype(float)
    t = (t.iloc[:,0])
    for m in mets:
        tdf.loc[t == 'mean',m] = (tdf['dev_{}'.format(m)]*100).round(2).astype(str) + " / " + (tdf['test_{}'.format(m)]*100).round(2).astype(str).astype(str)
        tdf.loc[t == 'std',m] = '(' + (tdf['dev_{}'.format(m)]*100).round(2).astype(str) + ") / (" + (tdf['test_{}'.format(m)]*100).round(2).astype(str) + ")"
        tdf.drop(['dev_{}'.format(m),'test_{}'.format(m)], axis = 1, inplace = True)

    print(tdf.columns)
    tdf = tdf.set_index(model)
    print(tdf.index)
    with open('artifacts/overall_perf.tex', 'w') as tf:
        tf.write(tdf.to_latex())




        # print(collected_metrics['confusion_matrix'][0].sum())

def make_per_label_perf(filepaths):
    """Make general performance table

    :filepath: Path to file

    """
    started = False
    filepaths.sort()
    filepaths = filepaths[2:4]
    path_dev = filepaths[0]
    path_test = filepaths[1]
    data_row = []

    all_datas = []
    for path,data_type in zip([path_dev,path_test],
                    ['dev','test']):
        data = read_pkl(path)
        key = list(data.keys())[0]
        datas = []
        for k in data.keys():
            datas.append(pd.DataFrame(
                pd.Series(data[k]['per_label']).values.tolist(),
            index= None).T.to_numpy())

        avg_data = np.array(datas).mean(axis =0)
        std_data = np.array(datas).std(axis = 0)
        avg_std_data = np.empty((avg_data.shape[0]+std_data.shape[0],
                                 avg_data.shape[1]))

        avg_std_data[::2,:] = avg_data
        avg_std_data[1::2,:] = std_data



        # Get percent composition of the labels
        cms = np.array([data[k]['weighted']['confusion_matrix'].sum(axis = 0)/data[k]['weighted']['confusion_matrix'].sum()
               for k in data.keys()])
        avg_cm = cms.mean(axis  = 0)
        std_cm = cms.std(axis  = 0)
        avg_std_cm = np.empty(avg_cm.shape[0]+std_cm.shape[0])

        avg_std_cm[::2,] = avg_cm
        avg_std_cm[1::2] = std_cm
        np.set_printoptions(precision = None, suppress = True)
        avg_std_data = np.round(np.concatenate([avg_std_data,avg_std_cm.reshape(20,-1)], axis = 1)*100, 2)

        all_datas.append(avg_std_data)

    avg_std_data = np.concatenate(all_datas,axis = 1)
    # print(avg_std_data)

    # Now let's make latex table
    mets = ['Precision','Recall','F1','Composition %']
    cols = ['dev_' + p for p in mets] +['test_' + p for p in mets]
    df = pd.DataFrame(avg_std_data, columns = cols)
    print(df.head())


    df.reset_index(inplace = True, drop = True)
    tdf = df
    tdf = tdf.astype(float)
    # tdf = tdf.round(2)
    # tdf[:,tdf.columns != 'type'] =tdf[:,tdf.columns != 'type'].astype(float)
    for m in mets:
        tdf.loc[range(0,20,2),m] = tdf['dev_{}'.format(m)].round(2).astype(str) + " / " + tdf['test_{}'.format(m)].round(2).astype(str).astype(str)
        tdf.loc[range(1,20,2),m] = "(" + tdf['dev_{}'.format(m)].round(2).astype(str) + ") / (" + tdf['test_{}'.format(m)].round(2).astype(str).astype(str) + ")"
        tdf.drop(['dev_{}'.format(m),'test_{}'.format(m)], axis = 1, inplace = True)

    # print(tdf.columns)
    print(tdf.head())
    labels =['caution_and_advice', 'displaced_people_and_evacuations',
       'infrastructure_and_utility_damage', 'injured_or_dead_people',
       'missing_or_found_people', 'not_humanitarian',
       'other_relevant_information', 'requests_or_urgent_needs',
       'rescue_volunteering_or_donation_effort', 'sympathy_and_support']
    broad_labels = list(itertools.chain.from_iterable([[d,""] for d in labels]))
    print(broad_labels)
    print(tdf.shape)
    tdf['label'] = broad_labels
    tdf = tdf.set_index('label')
    print(tdf.head())
    with open('artifacts/per_label_perf.tex', 'w') as tf:
        tf.write(tdf.to_latex())




        # print(collected_metrics['confusion_matrix'][0].sum())


def make_per_crisis_table(filepaths):
    """Make general performance table

    :filepath: Path to file

    """
    started = False
    filepaths.sort()
    test_paths = filepaths[3]
    dev_paths = filepaths[2]
    for path in [dev_paths,test_paths]:
            data = read_pkl(path)
            crisis_mets =\
                [data[k]['per_crisis']['weighted']
                  for k in data.keys()]


            mets = ['precision','recall','accuracy','F1']

            res_dict = {}
            for m in mets:
                for i in range(len(crisis_mets)):
                    for j in crisis_mets[i].keys():
                        if res_dict.get(j,None) is None:
                            res_dict[j] = {}
                        if res_dict[j].get(m,None) is None:
                            res_dict[j][m] = []

                        res_dict[j][m].append(crisis_mets[i][j][m])

            # key  ='california_wildfires_2018'
            avg_dict = {}
            std_dict = {}
            for c in res_dict.keys():
                avg_dict[c] = {}
                std_dict[c] = {}
                for m in res_dict[c].keys():
                    avg_dict[c][m] = np.array(res_dict[c][m]).mean()
                    std_dict[c][m] = np.array(res_dict[c][m]).std()


            avg = pd.DataFrame(avg_dict).T
            avg = avg[(~avg['accuracy'].isnull()) & (avg.index !='cyclone_idai_2019')]
            raw_ind = avg.index.tolist()
            avg = avg.to_numpy()
            ind = list(itertools.chain.from_iterable([[c[:-5],""] for c in raw_ind]))
            print(ind)
            year = list(itertools.chain.from_iterable([[c.split("_")[-1]]*2 for c in raw_ind]))
            print(year)
            std = pd.DataFrame(std_dict).T
            std = std[(~std['accuracy'].isnull()) & (std.index !='cyclone_idai_2019')].to_numpy()

            avg_std = np.empty((avg.shape[0]+std.shape[0],
                                    avg.shape[1]))

            avg_std[::2,:] = avg
            avg_std[1::2,:] = std
            df = pd.DataFrame(avg_std, columns = mets)
            df['ind'] = ind
            df = df.set_index('ind')
            df['year'] = year
            # df = df.sort_values(by = ['year','F1'])


            if df.shape[0] > 15:
                with open('artifacts/dev_crisis_perf.tex', 'w') as tf:
                    tf.write(df.to_latex())
            else:
                with open('artifacts/test_crisis_perf.tex', 'w') as tf:
                    tf.write(df.to_latex())




        # print(collected_metrics['confusion_matrix'][0].sum())
def read_pkl(filepath,
             root = 'artifacts/'):
    with open(filepath, 'rb') as file:
        return pkl.load(file)


#################################################################################
#   Main Method
#################################################################################

if __name__ == "__main__":
    root = 'artifacts/'
    files = glob(root + '*t100*.pkl')
    # print(files)
    # make_perf_table(files)
    # make_per_label_perf(files)
    make_per_crisis_table(files)
    # dic = read_pkl('



