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
from tqdm import tqdm
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
                 co_occ_range = [2,6],
                 simulate = True,
                 tokenizer = CountVectorizer().build_tokenizer(),
                 val_split = 0.2,
                 ):

        """Get data source locations and read in files"""
        self.train_path = train_path
        self.test_path = test_path

        self.train_dirs = glob(self.train_path + "*/*.tsv" )
        self.test_dirs = glob(self.test_path + "*/*.tsv" )

        self.tokenizer = tokenizer
        self.val_split = val_split

        self.co_occ_range = co_occ_range

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

    def __make_crisis_dict(self, data, crises):
        """Make train data dictionary by
        crisis

        :returns: data dictionary

        """
        data_dict = {}
        for c in crises:
            data_dict[c] = data[data['crisis'] == c].reset_index(drop=True)

        return data_dict

    def load_files(self, random_seed = 42):
        """Load files from directory

        :returns: Load file data from directory

        """

        # Load train data from specific directories
        # Split train data into train and dev sets
        self.train_corpus, self.dev_corpus = train_test_split(self.__load_file(self.train_dirs),
                                                              test_size = self.val_split,
                                                              random_state = random_seed)

        #Load test data
        self.test_corpus = self.__load_file(self.test_dirs)

        #Make label encoder
        self.le = LabelEncoder().fit(self.train_corpus['class_label'].drop_duplicates())

        #Get list of crises in each set
        self.train_crises = self.train_corpus['crisis'].drop_duplicates().to_numpy()
        self.dev_crises = self.train_corpus['crisis'].drop_duplicates().to_numpy()

        # Make sure that the same crises are present in train and dev data
        assert np.equal(self.train_crises.sort(), self.dev_crises.sort()).all(),\
            "Error, train and dev crisis mismatched - check information"

        # Get test crises
        self.test_crises = self.test_corpus['crisis'].drop_duplicates().to_numpy()

        #Make dictionary of crises
        self.train_dict = self.__make_crisis_dict(self.train_corpus, self.train_crises)
        self.dev_dict = self.__make_crisis_dict(self.dev_dict, self.dev_crises)
        self.test_dict = self.__make_crisis_dict(self.test_dict, self.test_crises)

    def tokenize(self, sentence):
        """Tokenize input based on provided tokenizer
        For now, just use the CountVect tokenizer
        until we get something more sophisticated

        :returns: Lists of tokens for embedding and encoding

        """
        return self.tokenizer(sentence)

    def next_batch(self, batch_size = 64):
        """Create next batch of data for synthesized training

        :batch_size: size of training batch
        :returns: batch of data-augmented multi-label samples

        """

        # Get crisis tokens and labels
        crisis_tokens, crisis_labels = self.__synthesize_data(batch_size)

        #Make true labels
        final_tokens_labels = [[],[]]
        for i,l in enumerate(crisis_labels):
            s = crisis_tokens[i]
            new_label = [np.ones(len(sentence))*self.le.transform([label])[0]
                         for sentence,label in zip(s, l)]

            #Chain together tokens and broadcasted labels
            chained_tokens = list(itertools.chain(*s))
            chained_labels = list(itertools.chain(*new_label))

            #Ensure tokens and labels are the same shape
            assert len(chained_tokens) == len(chained_labels),\
                "ERROR: train-label size mismatch in data augmentation engine"

            #Add sample to batch
            final_tokens_labels[0].append(chained_tokens)
            final_tokens_labels[1].append(chained_labels)

        return final_tokens_labels

    def __synthesize_data(self,batch_size):
        """ Synthesize multilabel text dataset
        and labels

        :returns: Synthesized train and test dataset

        """

        # Get varying sample sizes
        sample_sizes = np.random.randint(self.co_occ_range[0],
                                         self.co_occ_range[1],
                                         size = batch_size,
                                         dtype = int)

        #Sample only from specific crises
        sample_crises = np.random.choice(self.train_crises,size = batch_size)

        #Get specific random samples from every crisis
        crisis_inds = [
            np.random.randint(0, self.train_dict[c].shape[0],size = s)
            for c,s in zip(sample_crises, sample_sizes)]

        # Get a set of tuples joining sentences and labels together.
        # These are unexpanded and are sent compressed to next_batch
        # for expansion
        crisis_tokens_labels = list(zip(*
            [([self.tokenizer(sentence) for sentence
               in self.train_dict[c].loc[s,"tweet_text"].tolist()],
             self.train_dict[c].loc[s,"class_label"].tolist()) for c,s in
            zip(sample_crises, crisis_inds)
            ]))

        # Separate crisis tokens and labels
        crisis_tokens = crisis_tokens_labels[0]
        crisis_labels = crisis_tokens_labels[1]

        return crisis_tokens, crisis_labels

#######################################################################
# Testing
#######################################################################

if __name__ == "__main__":
    l = Loader()
    # print(l.train_dirs )
    l.load_files()
    # print(l.train_corpus.columns)
    # print(l.test_corpus.shape)
    # print(l.train_corpus.shape)
    # print(l.dev_corpus.shape)
    # print(l.dev_crises)
    # print(l.train_dict.keys())
    for i in tqdm(range(1000)):
        l.next_batch(batch_size = 64)
