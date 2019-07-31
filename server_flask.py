import os
import pickle
from flask import Flask, render_template, request
from translator import Translator
from keras.backend import clear_session

from utils import S3Bucket


# model_prefs.pkl
# 'model_path': 'models/de_to_en/dev_test_500/model.h5',
# 'source_tokenizer': obj,
# 'source_max_length': 5,
# 'source_vocab_size': 395,
# 'target_tokenizer': objk,
# 'target_vocab_size': 199,
# 'target_max_length': 3,
# 'total_count': 195847,
# 'train_count': 450,
# 'test_count': 50,
# 'BLEU1': 0.35171214694377334,
# 'BLEU2': 0.2028229102639773,
# 'BLEU3': 0.059549619334633,
# 'BLEU4': 1.1489659452541084e-78}

# Session scope variables
app = Flask(__name__)
model_id = 'basic_50K_35E/'
lang_prefix = {'French':'fr_to_en/',
                'German':'de_to_en/',
                'Italian':'it_to_en/',
                'Spanish':'es_to_en/',
                'Turkish': 'tr_to_en/'}

lang_options = [ {"Label": "Deutsch", "Value": "German", "Selected": False},
            {"Label": "Français", "Value": "French", "Selected": True},
            {"Label": "Italiano", "Value": "Italian", "Selected": False},
            {"Label": "Türk", "Value": "Turkish", "Selected": False},
            {"Label": "Español", "Value": "Spanish", "Selected": False}]

lang_index = 'French'

s3 = S3Bucket()

def get_selected(options):
    for option in options:
        if option["Selected"]:
            return option["Value"]

def set_language(lang_index):
    for option in lang_options:
        option["Selected"] = True if option["Value"] == lang_index else False

# HTML methods
@app.route('/')
def home_screen():
    return render_template('index.html', translation='', options=lang_options, selected_lang=get_selected(lang_options))

@app.route('/result',methods = ['POST', 'GET'])
def translate():
    if request.method == 'POST':

        # Get the results from the web user
        form_data = request.form
        for key, value in form_data.items():
            if key == 'Input_Text':
                input = value
                continue
            if key == 'Language':
                lang_index = value

        set_language(lang_index)

        # Get the model preferences locally or from S3
        s3_file = False

        try:
            if (s3_file):
                model_pref_path = 'machine-learning/models/' + lang_prefix[lang_index] + model_id + 'pickles/model_prefs.pkl'
                s3 = S3Bucket()
                model_prefs = pickle.load(s3.read_pickle(model_pref_path))
            else:
                model_pref_path = 'models/' + lang_prefix[lang_index] + model_id + 'pickles/model_prefs.pkl'
                model_prefs = pickle.load(open(model_pref_path, 'rb'))

        except Exception as e:
            input = e
            translation_error = 'No Model found for {}'.format(model_pref_path)
            return render_template('index.html',
                                    input_echo=input,
                                    input_text='Unable to load language model: ' + lang_index,
                                    translation=translation_error,
                                    selected_lang=get_selected(lang_options),
                                    options=lang_options)

        # A model exists, so use it and translate away!
        T = Translator(model_prefs)
        translation = T.translate(input)
        #
        # # Keras backend needs to clear the session
        clear_session()
        return render_template('index.html',
                                input_echo=input,
                                input_text=input,
                                translation=translation,
                                selected_lang=get_selected(lang_options),
                                options=lang_options)

        # for option in options:
        #     option["Selected"] = True if option["Value"] == lang_index else False
        # return render_template('index.html', input_text=input, translation=translation, selected_lang=get_selected(options), options=options)

if __name__ == '__main__':
    app.run(debug=True)
