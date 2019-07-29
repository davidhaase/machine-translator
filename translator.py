import re
import string
import pickle

from unicodedata import normalize

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

class Model():
    def __init__(self):
        pass

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
            self.model = load_model('models/de_to_en/model.h5')
            self.input_text = None

        except Exception as e:
            print(e)

    def encode_line(self, line):
        self.input_text = line
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
        line =np.array([' '.join(line)])
        # integer encode sequences
        line = self.preferences['source_tokenizer'].texts_to_sequences(line)
        # pad sequences with 0 values
        line = pad_sequences(line, maxlen=self.preferences['source_max_length'], padding='post')
        return line

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def translate(self, line):
        prediction = self.model.predict(self.encode_line(line), verbose=0)[0]
        integers = [np.argmax(vector) for vector in prediction]
        target = list()
        for i in integers:
            word = self.word_for_id(i, self.preferences['target_tokenizer'])
            if word is None:
                break
            target.append(word)
        return ' '.join(target)



if __name__ == '__main__':
    languages = {'Français': {'s3_file':'LanguageTexts/deu.txt', 'source':'de', 'target':'en', 'model_path':'models/fr_to_en/pickles/model_prefs.pkl'},
                'Deutsch': {'s3_file':'LanguageTexts/deu.txt', 'source':'fr', 'target':'en', 'model_path':'models/de_to_en/pickles/model_prefs.pkl'},
                'Italiano': {'s3_file':'LanguageTexts/deu.txt', 'source':'it', 'target':'en', 'model_path':'models/it_to_en/pickles/model_prefs.pkl'},
                'Español': {'s3_file':'LanguageTexts/deu.txt', 'source':'es', 'target':'en', 'model_path':'models/es_to_en/mpickles/model_prefs.pkl'}}

    lang = 'Deutsch'
    model_pref_path = languages[lang]['model_path']
    T = Translator(model_pref_path)

    de_string = 'Ich will nach Hause zuruk gehen'
    print(T.translate(de_string))
