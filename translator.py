import sys
import os
import re
import string
import pickle

from unicodedata import normalize

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import corpus_bleu

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.backend import clear_session

from utils import S3Bucket

s3 = S3Bucket()


def clean_line(line):
    table = str.maketrans('', '', string.punctuation)
    re_print = re.compile('[^%s]' % re.escape(string.printable))

    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    # tokenize on white space
    line = line.split()
    # convert to lowercase
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(table) for word in line]
    # remove non-printable chars form each token
    line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]
    # store as string
    return ' '.join(line)

def encode_lines(tokenizer, max_length, lines):
    # line = np.array([clean_line(line)])
    # integer encode sequences
    lines = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    lines = pad_sequences(lines, maxlen=max_length, padding='post')
    return lines




class Model():
    def __init__(self, language, model_name, description, force_rebuild=False):

        if model_name[-1] != '/':
            model_name += '/'
        cache_path = language['path'] + model_name
        if (os.path.isdir(cache_path)) and (force_rebuild==False):
            print('Model already exists at location {}'.format(cache_path))
            print('To overwrite existing model, pass \'force_rebuild=True\'')
            return None

        try:
            pickle_path = cache_path + 'pickles/'
            if not os.path.isdir(pickle_path):
                os.makedirs(pickle_path)

            image_path = cache_path + 'images/'
            if not os.path.isdir(image_path):
                os.makedirs(image_path)
        except Exception as e:
            print(e)
            return

        self.model_prefs = {'description':description}
        self.prefix = language['prefix']
        self.cache_path = cache_path
        self.model_path = cache_path + 'model.h5'
        self.source_file = language['s3_file']
        self.pickle_path = pickle_path
        self.image_path = image_path
        self.model = None
        self.source_lang= language['name']
        self.source_tokenizer = Tokenizer()
        self.source_vocab_size = 0
        self.source_max_length = 0
        self.target_lang = 'English'
        self.target_tokenizer = Tokenizer()
        self.target_vocab_size = 0
        self.target_max_length = 0
        self.raw_data = []
        self.subset = 0
        self.dataset = []
        self.dataset_file = pickle_path + 'dataset.pkl'
        self.clean_data = []
        self.clean_data_file = pickle_path + 'sentence_pairs.pkl'
        self.train = []
        self.train_file = pickle_path + 'train.pkl'
        self.test = []
        self.test_file = pickle_path + 'test.pkl'
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.text_y = None


    def get_data(self, subset, source_file='', s3=True):
        if (source_file != ''):
            self.source_file = source_file
        self.load_data(subset=subset, s3=s3)
        self.encode()

    def load_data(self, subset, s3):

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
        is_dataset = os.path.isfile(self.dataset_file)

        # But only load if ALL of the pickles are available
        # If any of the files are missing, rebuild them all
        if is_all and is_train and is_test and is_dataset:
            # There is still a pickle file locally, so reload that into memory
            self.clean_data = pickle.load(open(self.clean_data_file, 'rb'))
            print('Loaded {} sentences from existing file: {}'.format(str(len(self.clean_data)), self.clean_data_file))

            # There is still a pickle file locally, so reload that into memory
            self.train = pickle.load(open(self.train_file, 'rb'))
            print('Loaded {} sentences from train file: {}'.format(str(len(self.train)), self.train_file))

            # There is still a pickle file locally, so reload that into memory
            self.test = pickle.load(open(self.test_file, 'rb'))
            print('Loaded {} sentences from test file: {}'.format(str(len(self.test)), self.test_file))

            # There is still a pickle file locally, so reload that into memory
            self.dataset = pickle.load(open(self.dataset_file, 'rb'))
            print('Loaded {} sentences from test file: {}'.format(str(len(self.dataset)), self.dataset_file))

            # OK, you're done.  You've loaded it from pickles, so get out of here
            return


        print('Rebuilding data from {}'.format(self.source_file))

        if s3:
            s3=S3Bucket()
            self.raw_data = s3.read(self.source_file)
        else:
            f = open(self.source_file, 'rb')
            self.raw_data = f.read()
        self.clean_pairs()

        self.split_data(subset=subset)
        try:
            pickle.dump(self.clean_data, open(self.clean_data_file , 'wb+'))
            print('Saving all sentences {} to file: {}'.format(str(len(self.clean_data)), self.clean_data_file))
            pickle.dump(self.train, open(self.train_file, 'wb+'))
            print('Saving {} train sentences to file: {}'.format(str(len(self.train)), self.train_file))
            pickle.dump(self.test, open(self.test_file , 'wb+'))
            print('Saving {} test sentences to file: {}'.format(str(len(self.test)), self.test_file))
            pickle.dump(self.dataset, open(self.dataset_file , 'wb+'))
            print('Saving {} dataset sentences to file: {}'.format(str(len(self.dataset)), self.dataset_file))

        except Exception as e:
            print(e)
        return

    def clean_pairs(self):
        # split into (source, target) language tuples
        lines = self.raw_data.strip().split('\n')
        pairs = [line.split('\t') for line in lines]
        print(len(pairs))
        cleaned = list()
        for pair in pairs:
            clean_pair = list()
            for line in pair:
                clean_pair.append(clean_line(line))
            cleaned.append(clean_pair)
        self.clean_data = np.array(cleaned)

        return None

    def split_data(self, subset, split=90):
        assert ((len(self.clean_data) > 0) and (subset <= len(self.clean_data)))
        assert (split > 0 and split <= 100)

        slice_value = int((split/100)*subset)
        self.dataset = self.clean_data[:subset, :]


        # random shuffle
        np.random.shuffle(self.dataset)

        # split into train/test
        self.train = self.dataset[:slice_value]
        self.test = self.dataset[slice_value:]

        return None

    def encode_output(self, sequences, vocab_size):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = np.array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y

    def encode(self):
        # Be careful here!  Be sure to pick the correct tuple index for the language you want
        # German is the second value in the sentence tuples
        source_sentences = self.dataset[:, 1]
        # English is the first
        target_sentences = self.dataset[:, 0]

        self.source_tokenizer.fit_on_texts(source_sentences)
        self.source_vocab_size = len(self.source_tokenizer.word_index) + 1
        self.source_max_length = max(len(line.split()) for line in source_sentences)
        print('Source Vocabulary Size: {}'.format(str(self.source_vocab_size)))
        print('Source Max Sentence Length: {}'.format(str(self.source_max_length)))

        self.target_tokenizer.fit_on_texts(target_sentences)
        self.target_vocab_size = len(self.target_tokenizer.word_index) + 1
        self.target_max_length = max(len(line.split()) for line in target_sentences)
        print('Target Vocabulary Size: {}'.format(str(self.target_vocab_size)))
        print('Target Max Sentence Length: {}'.format(str(self.target_max_length)))

        # prepare training data
        self.train_X = encode_lines(self.source_tokenizer, self.source_max_length, self.train[:, 1])
        self.train_y = encode_lines(self.target_tokenizer, self.target_max_length, self.train[:, 0])
        self.train_y = self.encode_output(self.train_y, self.target_vocab_size)
        # # prepare validation data
        self.test_X = encode_lines(self.source_tokenizer, self.source_max_length, self.train[:, 1])
        self.test_y = encode_lines(self.target_tokenizer, self.target_max_length, self.train[:, 0])
        self.test_y = self.encode_output(self.test_y, self.target_vocab_size)

        # self.model_prefs = {'model_path': self.model_path,
        #                 'source_tokenizer':self.source_tokenizer,
        #                 'source_max_length':self.source_max_length,
        #                 'source_vocab_size': self.source_vocab_size,
        #                 'target_tokenizer':self.target_tokenizer,
        #                 'target_vocab_size':self.target_vocab_size ,
        #                 'target_max_length':self.target_max_length,
        #                 'total_count': len(self.clean_data),
        #                 'train_count': len(self.train),
        #                 'test_count': len(self.test)
        #                 }
        # try:
        #     pickle.dump(self.model_prefs, open(self.pickle_path + 'model_prefs.pkl', 'wb+'))
        # except Exception as e:
        #     print(e)

    def define_model(self, src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
        model.add(LSTM(n_units))
        model.add(RepeatVector(tar_timesteps))
        model.add(LSTM(n_units, return_sequences=True))
        model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
        return model

    def build_model(self, epochs=5):
        # define model
        model = self.define_model(self.source_vocab_size, self.target_vocab_size, self.source_max_length, self.target_max_length, 256)
        model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

        # summarize defined model
        print(model.summary())
        plot_model(model, to_file=self.image_path + 'model.png', show_shapes=True)

        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        acc = []
        val_acc = []
        loss = []
        val_loss = []

        total = len(self.train_X)
        start = 0
        interval = 10000
        end = start + interval
        steps = total/interval
        step = 0
        while end < total:
            start = end
            end += interval
            step += 1
            print('Step {} of {}'.format(str(step), str(steps)))

            train_X = self.train_X[start:end]
            train_y = self.train_y[start:end]
            test_X = self.test_X[start:end]
            test_y = self.test_y[start:end]
            history = model.fit(train_X, train_y, epochs=epochs, batch_size=64, validation_data=(test_X, test_y), callbacks=[checkpoint], verbose=2)
            acc += list(history.history['acc'])
            val_acc += list(history.history['val_acc'])
            loss += list(history.history['loss'])
            val_loss += list(history.history['val_loss'])
            self.save_figure(acc, val_acc, loss, val_loss)

        train_X = self.train_X[start:total]
        train_y = self.train_y[start:total]
        test_X = self.test_X[start:total]
        test_y = self.test_y[start:total]
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=64, validation_data=(test_X, test_y), callbacks=[checkpoint], verbose=2)
        acc += list(history.history['acc'])
        val_acc += list(history.history['val_acc'])
        loss += list(history.history['loss'])
        val_loss += list(history.history['val_loss'])
        self.save_figure(acc, val_acc, loss, val_loss)
        self.model = model

        blue1, blue2, blue3, blue4 = evaluate_model(model, self.target_tokenizer, self.train_X, self.train)

        self.model_prefs = {'model_path': self.model_path,
                        'source_tokenizer':self.source_tokenizer,
                        'source_max_length':self.source_max_length,
                        'source_vocab_size': self.source_vocab_size,
                        'target_tokenizer':self.target_tokenizer,
                        'target_vocab_size':self.target_vocab_size ,
                        'target_max_length':self.target_max_length,
                        'total_count': len(self.clean_data),
                        'train_count': len(self.train),
                        'test_count': len(self.test),
                        'BLEU1':blue1,
                        'BLEU2':blue2,
                        'BLEU3':blue3,
                        'BLEU4':blue4
                        }
        try:
            pickle.dump(self.model_prefs, open(self.pickle_path + 'model_prefs.pkl', 'wb+'))
        except Exception as e:
            print(e)
    def save_figure(self, acc, val_acc, loss, val_loss):
        plt.figure(figsize=(18, 10))
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('model_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig(self.image_path + 'accuracy.png')
        # plt.show()

        plt.figure(figsize=(18, 10))
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig(self.image_path + 'loss.png')
        # plt.show()

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src={}, target={}, predicted={}'.format(raw_src, raw_target, translation))
        actual.append([raw_target.split()])
        predicted.append(translation.split())
    # calculate BLEU score
    blue1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    blue2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    blue3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    blue4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: {}'.format(str(blue1)))
    print('BLEU-2: {}'.format(str(blue2)))
    print('BLEU-3: {}'.format(str(blue3)))
    print('BLEU-4: {}'.format(str(blue4)))
    return blue1, blue2, blue3, blue4

class Translator():
    def __init__(self, model_pref_path):
        try:
            self.preferences = pickle.load(open(model_pref_path, 'rb'))
            #dict.preferences = {'model_path': '',
            #                   'source_tokenizer': keras_obj,
            #                   'source_max_length': int,
            #                   'source_word_count': int,
            #                   'target_tokenizer': keras_obj,
            #                   'target_word_count': int,
            #                   'target_max_length': int }
            print(self.preferences['model_path'])
            self.model = load_model(self.preferences['model_path'])
            self.input_text = None

        except Exception as e:
            print(e)

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def translate(self, line):
        self.input_text = line
        tokenizer = self.preferences['source_tokenizer']
        maxlen = self.preferences['source_max_length']
        encoded_line = encode_lines(tokenizer, maxlen, np.array([clean_line(line)]))
        prediction = self.model.predict(encoded_line, verbose=0)[0]
        integers = [np.argmax(vector) for vector in prediction]
        target = list()
        for i in integers:
            word = self.word_for_id(i, self.preferences['target_tokenizer'])
            if word is None:
                break
            target.append(word)
        return ' '.join(target)



if __name__ == '__main__':
    # language = sys.argv[1]
    # model_name=sys.argv[2]
    # epochs=sys.argv[3] if len(sys.argv)>3 else 5
    # subset=int(sys.argv[4]) if len(sys.argv)>4 else 10000
    # force_rebuild=sys.argv[5] if len(sys.argv)>5 else False

    # languages = {'French' :{'name':'Français', 's3_file':'LanguageTexts/fra.txt', 'prefix': 'fr_to_en', 'path':'models/fr_to_en/'},
    #             'German' : {'name':'Deutsch', 's3_file':'LanguageTexts/deu.txt', 'prefix': 'de_to_en', 'path':'models/de_to_en/'},
    #             'Italian' : {'name':'Italiano', 's3_file':'LanguageTexts/ita.txt', 'prefix': 'it_to_en','path':'models/it_to_en/'},
    #             'Spanish' : {'name':'Español', 's3_file':'LanguageTexts/esp.txt', 'prefix': 'es_to_en','path':'models/es_to_en/'},
    #             'Danish' : {'name':'Español', 's3_file':'LanguageTexts/dan.txt', 'prefix': 'dk_to_en','path':'models/dk_to_en/'},
    #             'Turkish' : {'name':'Türk', 's3_file':'LanguageTexts/tur.txt', 'prefix': 'tr_to_en','path':'models/tr_to_en/'}}
    languages = {'Turkish' : {'name':'Türk', 's3_file':'LanguageTexts/tur.txt', 'prefix': 'tr_to_en','path':'models/tr_to_en/'}}

    subset = 50000
    description = 'Basic word clean-up; 50,000 sentences, 35 epochs of 10,000 batches'
    model_name = 'basic_50K_35E'
    epochs = 35


    for lang in languages:
        print('Processing {}'.format(lang))
        model = Model(language=languages[lang], model_name=model_name, description=description, force_rebuild=True)
        model.get_data(subset=subset)
        model.build_model(epochs=int(epochs))
        clear_session()
