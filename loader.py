#################################################################################
#
#             Project Title:  HumAID Data Loader              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-04-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import pandas as pd
import numpy as np
from glob import glob

#################################################################################
#   Function-Class Declaration
#################################################################################

class Loader(object):
    """
    Loads data for training, testing.

    Also owns dataset augmentation

    """

    def __init__(self,
                 train_path = "data/preq_2018/",
                 test_path = "data/post_2018/",
                 co_occ_min_max = [2,6],
                 mix_same_crisis = True,
                 dataset_size = 100000,
                 simulate = True,
                 ):
        """Instantiate id artifacts """
        self.train_path = train_path
        self.test_path = test_path

        self.train_dirs = glob(self.train_path + "*/*.tsv" )
        self.test_dirs = glob(self.test_path + "*/*.tsv" )

    def __load_file(self, flist):
        """Load files into pandas df with
        glob and a list of files. Assumes
        tab separation

        :flist: list of files
        :returns: Pandas dataframe of corpus

        """
        dfs = []
        for f in flist:
            df = pd.read_csv(f, sep = "\t")
            tag = "_".join(f.split("/")[-1].split("_")[:-1])
            df['crisis'] = tag
            dfs.append(df)
        return pd.concat(dfs)

    def load_files(self):
        """Load files from directory

        :returns: Load file data from directory

        """
        self.train_corpus = self.__load_file(self.train_dirs)
        self.test_corpus = self.__load_file(self.test_dirs)

    def preprocess_files(self):
        """Preprocess files with proper tokenization
        and-or embeddings.


        :returns: Preprocess corpus for modeling

        """
        pass

    def synthesize_data(self):
        """ Synthesize multilabel text dataset
        and labels

        :returns: Synthesized train and test dataset

        """
        pass


#######################################################################
# Testing
#######################################################################

if __name__ == "__main__":
    l = Loader()
    # print(l.train_dirs )
    l.load_files()
    print(l.test_corpus.head())
    print(l.train_corpus.head())

