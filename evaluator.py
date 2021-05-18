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


#################################################################################
#   Function-Class Declaration
#################################################################################

class CrisisEvaluator(object):

    """Evaluates Performance for Crises,
    Will make tables and graphs easier later"""

    def __init__(self, data_loader):
        """TODO: to be defined.

        :loader: Data loader
        """
        self.loader = data_loader
        assert self.loader.le is not None, "Error, No label encoder for data loader"
        self.le = self.loader.le
        self.perf_dict = {}

    def get_per_label_perf(self,
                           experiment_name,
                           preds, truth):
        """Get per label performance

        :experiment_name: Name of experiment
        :preds: Predictions
        :truth: Ground Truth

        """
        pass

    def get_per_crisis_perf(self,
                           experiment_name,
                           preds, truth):
        """Get per label performance

        :experiment_name: Name of experiment
        :preds: Predictions
        :truth: Ground Truth

        """
        pass
    def (self, arg1):
        """TODO: Docstring for .

        :arg1: TODO
        :returns: TODO

        """
        pass






#################################################################################
#   Main Method
#################################################################################



