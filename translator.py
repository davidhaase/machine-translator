import os
import string
import pickle
import re

import numpy as np

from unicodedata import normalize

from nltk.translate.bleu_score import corpus_bleu

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from utils import S3Bucket

class Model():
    def __init__(self, source_lang, target_lang):
        self.source_path = ''
        self.prefix = source_lang + '2' + target_lang
        self.source_lang= source_lang
        self.source_tokenizer = None
        self.source_word_count = 0
        self.source_max_length = 0
        self.target_lang = target_lang
        self.target_tokenizer = None
        self.target_word_count = 0
        self.target_max_length = 0
        self.raw_data = []
        self.dataset = []
        self.clean_data = []
        self.clean_data_file = 'pickles/' + self.prefix + '_sentence_pairs.pkl'
        self.train = []
        self.train_file = 'pickles/' + self.prefix + '_train.pkl'
        self.test = []
        self.test_file = 'pickles/' + self.prefix + '_test.pkl'
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.text_y = None
        self.table = str.maketrans('', '', string.punctuation)
        self.re_print = re.compile('[^%s]' % re.escape(string.printable))


    def get_data(self, source_path, s3=True):
        self.load_data(source_path, s3)
        self.split_data()
        self.encode()

    def load_data(self, source_path, s3=True):

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

        if s3:
            s3=S3Bucket()
            self.raw_data = s3.read(source_path)
        else:
            f = open(source_path, 'rb')
            self.raw_data = f.read()

        self._clean_pairs()
        return

    def clean_line(self, line):
        # normalize unicode characters
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        # tokenize on white space
        line = line.split()
        # convert to lowercase
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [word.translate(self.table) for word in line]
        # remove non-printable chars form each token
        line = [self.re_print.sub('', w) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        return ' '.join(line)

    def _clean_pairs(self):
        # split into (source, target) language tuples
        lines = self.raw_data.strip().split('\n')
        pairs = [line.split('\t') for line in lines]
        cleaned = list()
        for pair in pairs:
            clean_pair = list()
            for line in pair:
                clean_pair.append(self.clean_line(line))
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
        self.dataset = self.clean_data[:subset, :]

        # random shuffle
        np.random.shuffle(self.dataset)

        # split into train/test
        self.train = self.dataset[:slice_value]
        self.test = self.dataset[slice_value:]

        try:
            pickle.dump(self.train, open(self.train_file, 'wb+'))
            print('Saving {} train sentences to file: {}'.format(str(len(self.train)), self.train_file))
            pickle.dump(self.test, open(self.test_file, 'wb+'))
            print('Saving {} test sentences to file: {}'.format(str(len(self.test)), self.test_file))
        except Exception as e:
            print(e)

    def encode(self):
        # Be careful here!  Be sure to pick the correct tuple index for the language you want
        source_sentences = self.dataset[:, 1]
        target_sentences = self.dataset[:, 0]

        self.source_tokenizer = Tokenizer()
        self.source_tokenizer.fit_on_texts(source_sentences)
        self.source_word_count = len(self.source_tokenizer.word_index) + 1
        self.source_max_length = max(len(line.split()) for line in source_sentences)
        print('Source Vocabulary Size: {}'.format(str(self.source_word_count)))
        print('Source Max Sentence Length: {}'.format(str(self.source_max_length)))

        self.target_tokenizer = Tokenizer()
        self.target_tokenizer.fit_on_texts(target_sentences)
        self.target_word_count = len(self.target_tokenizer.word_index) + 1
        self.target_max_length = max(len(line.split()) for line in target_sentences)
        print('Target Vocabulary Size: {}'.format(str(self.target_word_count)))
        print('Target Max Sentence Length: {}'.format(str(self.target_max_length)))

        tokenizers = {'source':self.source_tokenizer, 'target':self.target_tokenizer}
        try:
            pickle.dump(tokenizers, open('pickles/tokenizers.pkl', 'wb+'))
        except Exception as e:
            print(e)

    def encode_source(self, lines):
        # integer encode sequences
        y = self.source_tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        y = pad_sequences(y, self.source_max_length, padding='post')
        y_list = list()
        for sequence in y:
            encoded = to_categorical(sequence, num_classes=self.source_word_count)
            y_list.append(encoded)
        y = np.array(y_list)
        y = y.reshape(y.shape[0], y.shape[1], self.source_word_count)
        print(y)



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
d2e_Data = Model('de', 'en')
d2e_Data.get_data(s3_file)
d2e_Data.encode_source('Guten Tag')
# d2e_Data.clean_pairs()

# line = 'Ich bin ein bischen unmüglich'
# print(d2e_Data.clean_line(line))
# d2e_Model = Model()
# d2e = Translator(d2e_Model)
#
#
# input_sentence = 'Hello, world! My name is David Haase. What is your name?'
# d2e.translate(input_sentence, show_bleu=False)
