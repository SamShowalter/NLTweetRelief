################################################################################
#
#             Project Title:  HumAID Data Loader              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-04-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
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
                 train_path = "data/train_val/",
                 test_path = "data/test/",
                 co_occ_range = [2,6],
                 tokenizer = CountVectorizer().build_tokenizer(),
                 val_split = 0.2,
                 random_seed =42,
                 ):

        """Get data source locations and read in files"""
        self.train_path = train_path
        self.test_path = test_path
        np.random.seed(random_seed)

        self.train_dirs = glob(self.train_path + "*/*.tsv" )
        self.test_dirs = glob(self.test_path + "*/*.tsv" )
        self.unilabel_df = None

        self.tokenizer = tokenizer
        self.val_split = val_split

        self.random_seed = random_seed

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
            df = pd.read_csv(f, sep = "\t", encoding='utf8')
            df['tweet_text'] = df['tweet_text'].apply(lambda x: x.encode('utf8').decode('latin-1', 'ignore'))
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

    def preprocess_unilabel(self, sample, label):
        sentence = self.tokenize(sample)
        labels = np.ones(len(sentence))*self.label_le.transform([label])
        return sentence, labels

    def __synthesize_data(self,batch_size, dataset="train"):
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
        _dict = self.train_dict
        sample_crises = np.random.choice(self.train_crises,size = batch_size)
        if dataset == "dev":
            _dict = self.dev_dict
            sample_crises = np.random.choice(self.dev_crises,size = batch_size)
        if dataset == "test":
            _dict = self.test_dict
            sample_crises = np.random.choice(self.test_crises,size = batch_size)


        #Get specific random samples from every crisis
        crisis_inds = [
            np.random.randint(0, _dict[c].shape[0],size = s)
            for c,s in zip(sample_crises, sample_sizes)]

        # Get a set of tuples joining sentences and labels together.
        # These are unexpanded and are sent compressed to next_batch
        # for expansion
        crisis_tokens_labels = list(zip(*
            [([self.tokenize(sentence) for sentence
               in _dict[c].loc[s,"tweet_text"]],
             _dict[c].loc[s,"class_label"]) for c,s in
            zip(sample_crises, crisis_inds)
            ]))

        # Separate crisis tokens and labels
        crisis_tokens = crisis_tokens_labels[0]
        crisis_labels = crisis_tokens_labels[1]

        return crisis_tokens, crisis_labels, self.crisis_le.transform(sample_crises)

    def load_files(self):
        """Load files from directory

        :returns: Load file data from directory

        """

        # Load train data from specific directories
        # Split train data into train and dev sets
        self.train_corpus, self.dev_corpus = train_test_split(self.__load_file(self.train_dirs),
                                                              test_size = self.val_split,
                                                              random_state = self.random_seed)

        #Load test data
        self.test_corpus = self.__load_file(self.test_dirs)

        #Make label encoder
        self.label_le = LabelEncoder().fit(self.train_corpus['class_label'].drop_duplicates())

        #Get list of crises in each set
        self.train_crises = self.train_corpus['crisis'].drop_duplicates().to_numpy()
        self.dev_crises = self.dev_corpus['crisis'].drop_duplicates().to_numpy()

        # Make sure that the same crises are present in train and dev data
        assert np.equal(self.train_crises.sort(), self.dev_crises.sort()).all(),\
            "Error, train and dev crisis mismatched - check information"

        # Get test crises
        self.test_crises = self.test_corpus['crisis'].drop_duplicates().to_numpy()

        #Encode crises
        self.crisis_le = LabelEncoder().fit(np.concatenate([self.train_crises,
                                                            self.dev_crises,
                                                            self.test_crises]))

        #Make dictionary of crises
        self.train_dict = self.__make_crisis_dict(self.train_corpus, self.train_crises)
        self.dev_dict = self.__make_crisis_dict(self.dev_corpus, self.dev_crises)
        self.test_dict = self.__make_crisis_dict(self.test_corpus, self.test_crises)

    def tokenize(self, sentence):
        """Tokenize input based on provided tokenizer
        For now, just use the CountVect tokenizer
        until we get something more sophisticated

        :returns: Lists of tokens for embedding and encoding

        """
        return self.tokenizer(sentence)

    def next_batch_multilabel(self, batch_size = 64, dataset="train"):
        """Create next batch of data for synthesized training

        :batch_size: size of training batch
        :returns: batch of data-augmented multi-label samples

        We are given crisis data for different events - there are tweets for each
        To do data augmentation:
            - Choose crisis randomly
            - Choose a random number of tweets in a range (2,6)
            - ['this', 'tweet'] -> [0,0]
            - Concatenate all tweets together to make a token-level dataset.
        """

        # Get crisis tokens and labels
        crisis_tokens, crisis_labels, crises = self.__synthesize_data(batch_size, dataset=dataset)

        #Make true labels
        final_tokens_labels = [[],[],[]]
        for i,l in enumerate(crisis_labels):
            s = crisis_tokens[i]
            c = crises[i]
            new_label = [(np.ones(len(sentence))*self.label_le.transform([label])[0]).astype(int)
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
            final_tokens_labels[2].append(c)

        return final_tokens_labels

    def next_epoch_multilabel(self, num_batches = 100,
                   batch_size = 64,
                   dataset="train",
                   verbose = True):
        """Create a set of new data_aug batches for
        experimentation and training

        :num_batches: number of batches in an epoch
        :returns: list of batches

        """

        if verbose:
            print("Creating {} multilabel batches of {} samples from {} set for next epoch"
              .format(num_batches,batch_size, dataset))
        epoch = []
        for i in tqdm(range(num_batches), disable = not verbose):
            epoch.append(self.next_batch_multilabel(batch_size = batch_size,
                        dataset=dataset))

        return epoch

    def next_epoch_unilabel(self, batch_size = 64,dataset="train",
                            verbose = True):
        """Get next epoch of data for unilabel dataset

        :returns: list of batches in certain size, plus
                  data is shuffled
        """

        corpus = self.train_corpus
        if dataset == "dev":
            corpus = self.dev_corpus
        if dataset == "test":
            corpus = self.test_corpus
        train_samples = corpus['tweet_text']\
                .apply(lambda x: self.tokenize(x)).reset_index(drop = True)

        train_labels_sentence = pd.DataFrame(self.label_le\
                                            .transform(corpus['class_label']),
                                            columns = ['label'])

        train_labels_sentence['sample_len'] = train_samples\
            .apply(lambda s: len(s)).reset_index(drop=True)

        token_labels = train_labels_sentence\
                .apply(lambda row: (np.ones(row['sample_len'])*row['label']).astype(int), axis = 1)

        # print(train_samples)
        # print(token_labels)
        self.unilabel_df = pd.concat([train_samples, token_labels], axis =1)

        shuffled_df = list(zip(*self.unilabel_df.sample(frac = 1).values.tolist()))

        batches = []
        if verbose:
            print("Preparing unilabel batches of {} samples taken from {} set for next epoch".format(batch_size, dataset))
        for i in tqdm(range(0,len(shuffled_df[0]), batch_size), disable = not verbose):
            batches.append([shuffled_df[0][i:i+batch_size],
                            shuffled_df[1][i:i+batch_size]])

        return batches

    def next_epoch(self,num_batches =100,
                   batch_size = 64,
                   simulate = True,
                   dataset = "train",
                   verbose = True):
        if simulate:
            return self.next_epoch_multilabel(num_batches,
                                              batch_size,
                                              dataset=dataset,
                                              verbose = verbose)
        else:
            return self.next_epoch_unilabel(batch_size,dataset=dataset,
                                            verbose = verbose)





#######################################################################
# Testing
#######################################################################

if __name__ == "__main__":
    l = Loader()
    l.load_files()
    # print(l.train_crises)

    # print(l.train_corpus.columns)
    print(l.test_corpus.shape)
    print(l.train_corpus.shape)
    print(l.dev_corpus.shape)
    print(l.test_crises)
    print(l.train_crises)
    assert len(set(l.train_crises) & set(l.test_crises)) == 0, "ERROR: Bleed from train to test set"

#     for k in l.train_dict.keys():
#         print("Train then dev shapes")
#         print(l.train_dict[k].shape)
#         print(l.dev_dict[k].shape)
    # print(l.dev_crises)
    # print(l.train_dict.keys())

    ##Visually inspect a sample
    #tokens_labels = l.next_batch(batch_size = 64)
    #sample = tokens_labels[0][0]
    #sample = [i.encode('cp1252') for i in sample]
    #print(sample)
    #print(tokens_labels[1][0])
    #print(l.label_le.inverse_transform(tokens_labels[1][0]))
    #sys.exit(1)

    # Test of creating an epoch
    # l.next_epoch()

    # d = l.next_epoch_unilabel(dataset = "test")
    # print(len(d))

    d = l.next_epoch_multilabel(dataset = "test")

