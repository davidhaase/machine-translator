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
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

from utils import S3Bucket

class Model():
    def __init__(self, source_lang, target_lang):
        prefix = source_lang + '_to_' + target_lang
        self.cache_path = 'models/' + prefix + '/'
        if not os.path.isdir(self.cache_path + 'pickles/'):
            os.makedirs(self.cache_path + 'pickles/')
        self.model = None
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
        self.clean_data_file = self.cache_path + 'pickles/sentence_pairs.pkl'
        self.train = []
        self.train_file = self.cache_path + 'pickles/train.pkl'
        self.test = []
        self.test_file = self.cache_path + 'pickles/test.pkl'
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
        self.split_data()

        try:
            pickle.dump(self.clean_data, open(self.clean_data_file , 'wb+'))
            print('Saving all sentences {} to file: {}'.format(str(len(self.clean_data)), self.clean_data_file))
            pickle.dump(self.train, open(self.train_file, 'wb+'))
            print('Saving {} train sentences to file: {}'.format(str(len(self.train)), self.train_file))
            pickle.dump(self.test, open(self.test_file , 'wb+'))
            print('Saving {} test sentences to file: {}'.format(str(len(self.test)), self.test_file))
        except Exception as e:
            print(e)
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

        return None

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

        model_prefs = {'model_path': self.source_path + 'model.h5',
                        'source_tokenizer':self.source_tokenizer,
                        'source_max_length':self.source_max_length,
                        'source_word_count': self.source_word_count,
                        'target_tokenizer':self.target_tokenizer,
                        'target_word_count':self.target_word_count ,
                        'target_max_length':self.target_max_length
                        }
        try:
            pickle.dump(model_prefs, open(self.cache_path + 'pickles/model_prefs.pkl', 'wb+'))
        except Exception as e:
            print(e)





class Translator():
    def __init__(self, souce_file, source_lang, target_lang):
        self.input_lines = ''
        self.Model = Model(source_lang, target_lang)
        self.Model.get_data(souce_file)

    def word_for_id(self, integer, tokenizer):
    	for word, index in tokenizer.word_index.items():
    		if index == integer:
    			return word
    	return None
    def translate(self, Input_Text, show_BLEU=True):

        predicted = Input_Text.upper()
        actual = Input_Text

        bleu_scores = []
        bleu_scores.append('BLEU-1: {}\n'.format(str(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))))
        bleu_scores.append('BLEU-2: {}\n'.format(str(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))))
        bleu_scores.append('BLEU-3: {}\n'.format(str(corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))))
        bleu_scores.append('BLEU-4: {}\n'.format(str(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))))
        translation = {'input_text':actual, 'translation':predicted}
        print(translation)
        print(bleu_scores)
        if show_BLEU:
            return (translation, bleu_scores)
        else:
            return (translation, '')

if __name__ == '__main__':
    s3_file = 'LanguageTexts/deu.txt'
    German = Translator(s3_file, 'de', 'en')
    German.translate('guten morgen')
# s3_file = 'LanguageTexts/deu.txt'
# d2e_Data = Model('de', 'en')
# d2e_Data.get_data(s3_file)
