import os
import pickle
import re

import numpy as np

from nltk.translate.bleu_score import corpus_bleu

from utils import S3Bucket, clean_line

table = str.maketrans('', '', string.punctuation)
re_print = re.compile('[^%s]' % re.escape(string.printable))

def clean_line(line):
    # normalize unicode characters
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    # tokenize on white space
    line = line.split()
    # convert to lowercase
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(table    ) for word in line]
    # remove non-printable chars form each token
    line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]
    # store as string
    return ' '.join(line)

class Data():
    def __init__(self, source_lang, target_lang):
        self.source_path = ''
        self.prefix = source_lang + '2' + target_lang
        self.source_lang= source_lang
        self.target_lang = target_lang
        self.raw_data = []
        self.clean_data = []
        self.clean_data_file = 'pickles/' + self.prefix + '_sentence_pairs.pkl'
        self.train = []
        self.train_file = 'pickles/' + self.prefix + '_train.pkl'
        self.test = []
        self.test_file = 'pickles/' + self.prefix + '_test.pkl'


    def load_data(self, source_path, use_s3=True):

        self.source_path = source_path

        # The data source being requested is the same that might already exist
        # So run through the following series of checks to see if data already exists locally
        if (len(self.raw_data) > 0) and (len(self.clean_data) > 0) and (len(self.train) > 0) and (len(self.test) > 0):
            # There is already data loaded, so use that
            print('{} lines of raw data already loaded.'.format(len(self.raw_data)))
            print('{} lines of clean data already loaded.'.format(len(self.clean_pairs)))
            print('\t{} lines of traning data already loaded.'.format(len(self.train)))
            print('\t{} lines of test data already loaded.'.format(len(self.test)))
            # Print some values and get out of here
            return

        # It didn't exist in memory, so try loading from pickle files
        is_all = os.path.isfile(self.clean_data_file)
        is_train = os.path.isfile(self.train_file)
        is_test = os.path.isfile(self.test_file)

        # But only load if ALL of the pickles are available
        # If any of the files are missing, rebuild them all
        if is_all and is_train and is_test:
            # There is still a pickle file locally, so reload that into memory
            self.clean_data = pickle.load(open(self.clean_data_file, 'rb'))
            print('Loaded {} sentences from existing file: {}'.format(str(len(self.clean_data)), self.clean_data_file))

            # There is still a pickle file locally, so reload that into memory
            self.train = pickle.load(open(self.train_file, 'rb'))
            print('Loaded {} sentences from train file: {}'.format(str(len(self.train)), self.train_file))

            # There is still a pickle file locally, so reload that into memory
            self.test = pickle.load(open(self.test_file, 'rb'))
            print('Loaded {} sentences from test file: {}'.format(str(len(self.test)), self.test_file))

            # OK, you're done.  You've loaded it from pickles, so get out of here
            return


        print('Rebuilding data from {}'.format(source_path))

        if use_s3:
            s3=S3Bucket()
            self.raw_data = s3.read(source_path)
        else:
            f = open(source_path, 'rb')
            self.raw_data = f.read()

        self._clean_pairs()
        return


    def _clean_pairs(self):
        # split into (source, target) language tuples
        lines = self.raw_data.strip().split('\n')
        pairs = [line.split('\t') for line in lines]
        cleaned = list()
        for pair in pairs:
            clean_pair = list()
            for line in pair:
                clean_pair.append(clean_line(line))
            cleaned.append(clean_pair)
        self.clean_data = np.array(cleaned)
        pickle.dump(self.clean_data, open(self.clean_data_file, 'wb+'))
        print('Saving all sentences {} to file: {}'.format(str(len(self.clean_data)), self.clean_data_file))

        self.split_data()

        return None

    def split_data(self, subset=10000, split=90):
        assert ((len(self.clean_data) > 0) and (len(self.clean_data) > subset))
        assert (split > 0 and split <= 100)

        slice_value = int((split/100)*subset)
        dataset = self.clean_data[:subset, :]
        # random shuffle
        np.random.shuffle(dataset)
        # split into train/test
        self.train = dataset[:slice_value]
        self.test = dataset[slice_value:]

        try:
            pickle.dump(self.train, open(self.train_file, 'wb+'))
            print('Saving {} train sentences to file: {}'.format(str(len(self.train)), self.train_file))
            pickle.dump(self.test, open(self.test_file, 'wb+'))
            print('Saving {} test sentences to file: {}'.format(str(len(self.test)), self.test_file))
        except Exception as e:
            print(e)

class Model():
    def __init__(self):
        self.model = None
        self.source_tokenizer = None
        self.target_tokenizer = None

    def _encode(self, sentence):
        sequence = sentence.upper()
        return sequence

    def build_model(self, Data, force_rebuild=False):
        pass

class Translator():
    def __init__(self, Model):
        self.Model = Model

    def translate(self, actual, show_bleu=True):
        predicted = self.Model._encode(actual)
        print('src = {}\npredicted = {}'.format(actual, predicted))
        if show_bleu:
            print('BLEU-1: {}'.format(str(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))))
            print('BLEU-2: {}'.format(str(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))))
            print('BLEU-3: {}'.format(str(corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))))
            print('BLEU-4: {}'.format(str(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))))
        return None


s3_file = 'LanguageTexts/deu.txt'
d2e_Data = Data('de', 'en')
d2e_Data.load_data(s3_file)
# d2e_Data.clean_pairs()

# line = 'Ich bin ein bischen unmüglich'
# print(d2e_Data.clean_line(line))
# d2e_Model = Model()
# d2e = Translator(d2e_Model)
#
#
# input_sentence = 'Hello, world! My name is David Haase. What is your name?'
# d2e.translate(input_sentence, show_bleu=False)
