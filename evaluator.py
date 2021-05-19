#################################################################################
#
#             Project Title:  Multilabel Performance Eval           Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-05-18
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import pandas as pd
import numpy as np
from utils import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import itertools

#################################################################################
#   Function-Class Declaration
#################################################################################

class CrisisEvaluator(object):

    """Evaluates Performance for Crises,
    Will make tables and graphs easier later"""

    def __init__(self, data_loader):
        """TODO: Evaluator for Crisis experiments

        :loader: Data loader
        """
        self.loader = data_loader
        assert self.loader.label_le is not None, "Error, No label encoder for data loader"
        assert self.loader.crisis_le is not None, "Error, No label encoder for data loader"
        self.label_le = self.loader.label_le
        self.crisis_le =self.loader.crisis_le

        # Performance dictionary
        self.perf_dict = {}

        #Replace this with real values eventually
        self.num_labels = len(self.label_le.classes_)
        self.num_crises = len(self.crisis_le.classes_)

        # Metrics to compute
        self.mets = {'confusion_matrix':confusion_matrix,
                     'precision':precision_score,
                     'recall':recall_score,
                     # 'accuracy':accuracy_score,
                     'F1':f1_score}

    def get_per_label_perf(self,
                           experiment_name,
                           pred_batches, label_batches):
        """Get per label performance

        :experiment_name: Name of experiment
        :pred_batches: Batches of only predictions
        :label_batches: original batches

        """

        if not self.perf_dict.get(experiment_name):
            self.perf_dict[experiment_name] = {}
        if not self.perf_dict[experiment_name].get("per_label"):
            self.perf_dict[experiment_name]['per_label'] = {}

        preds = np.array(list(itertools.chain.from_iterable(pred_batches)))
        labels_isolated = [b[1] for b in label_batches]
        labels = np.array(list(itertools.chain.from_iterable(labels_isolated)))

        for key,metric in self.mets.items():
            if key == 'confusion_matrix':
                self.perf_dict[experiment_name]['per_label'][key]\
                    = metric(labels, preds)
            else:
                self.perf_dict[experiment_name]['per_label'][key]\
                    = metric(labels, preds,
                             labels = range(0,self.label_len),
                            average = None)

    def get_perf(self,
                experiment_name,
                 pred_batches, label_batches,
                 kind = 'macro'):
        """Get performance metrics

        :experiment_name: Name of experiment
        :pred_batches: Batches of only predictions
        :label_batches: original batches

        """

        if not self.perf_dict.get(experiment_name):
            self.perf_dict[experiment_name] = {}
        if not self.perf_dict[experiment_name].get(kind):
            self.perf_dict[experiment_name][kind] = {}

        preds = np.array(list(itertools.chain.from_iterable(pred_batches)))
        labels_isolated = [b[1] for b in label_batches]
        labels = np.array(list(itertools.chain.from_iterable(labels_isolated)))

        for key,metric in self.mets.items():
            if key == 'confusion_matrix':
                continue
            else:
                self.perf_dict[experiment_name][kind][key]\
                    = metric(labels, preds,
                            average = kind)

    def get_per_crisis_perf(self,
                experiment_name,
                 pred_batches, label_batches,
                 kind = 'macro'):
        """Get per label performance

        :experiment_name: Name of experiment
        :preds: Predictions
        :truth: Ground Truth

        """

        if not self.perf_dict.get(experiment_name):
            self.perf_dict[experiment_name] = {}
        if not self.perf_dict[experiment_name].get('per_crisis'):
            self.perf_dict[experiment_name]['per_crisis'] = {}
        if not self.perf_dict[experiment_name]['per_crisis'].get(kind):
            self.perf_dict[experiment_name]['per_crisis'][kind] = {}

        # Isolate preds, labels, crises
        preds = np.array(list(itertools.chain.from_iterable(pred_batches)))
        labels_isolated = [b[1] for b in label_batches]
        crises_isolated = [[b[2]]*len(b[1]) for b in label_batches]
        labels = np.array(list(itertools.chain.from_iterable(labels_isolated)))
        crises = np.array(list(itertools.chain.from_iterable(crises_isolated)))

        #Isolate crisis names and iterate to get performance
        c_names = self.crisis_le.inverse_transform(range(0,self.num_crises))
        for c in range(self.num_crises):
            self.perf_dict[experiment_name]['per_crisis'][kind][c_names[c]] = {}
            crisis_mask = crises == c

            crisis_preds = preds[crisis_mask]
            crisis_labels = labels[crisis_mask]

            for key,metric in self.mets.items():
                if key == 'confusion_matrix':
                    continue
                else:
                    self.perf_dict[experiment_name]['per_crisis'][kind][c_names[c]][key]\
                        = metric(crisis_labels, crisis_preds,
                                average = kind)

#######################################################################
# Main method
#######################################################################

if __name__ == "__main__":
#     preds = [ [1,2,3,1,0,0,0,0,0],
#             [1,2,0,1,0,2,3,0,1]]

#     batch_labels =[ [1,1,3,3,2,0,0,2,1],
#             [2,2,0,1,0,0,3,0,1]]

#     batch_data =[ [0,0,0,0,2,0,3,2,1],
#             [1,2,0,1,0,2,3,0,1]]

#     batch_crises =["crisis_0", "crisis_1"]
#     crisis_le = LabelEncoder().fit(batch_crises)
#     batch_crises = crisis_le.transform(batch_crises)
#     batch = list(zip(batch_data, batch_labels, batch_crises))


    # ce.get_per_crisis_perf("Test_experiment",preds, batch, kind = 'micro')
    # ce.get_per_crisis_perf("Test_experiment",preds, batch, kind = 'macro')

    # ce.get_per_label_perf("Test_experiment",preds, batch)
    # ce.get_perf("Test_experiment",preds, batch, kind = 'micro')
    # ce.get_perf("Test_experiment",preds, batch, kind = 'macro')
    # ce.get_perf("Test_experiment",preds, batch, kind = 'weighted')
    # print(ce.perf_dict)

    from loader import Loader

    l = Loader()
    l.load_files()
    ce = CrisisEvaluator(l)
    print(ce.num_crises)
    print(ce.num_labels)


