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
model_id = 'basic_75K_35E_fixed/'
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

bleus = {
    "French" : "1-grams: 0.6588</br>bi-grams: 0.5447</br>tri-grams: 0.4940</br>4-grams: 0.3815</br><hr>loss: 0.2570</br>acc: 0.9801</br>val_loss: 2.1694</br>val_acc: 0.7039",
    "German" : "1-grams: 0.6703</br>bi-grams: 0.5568</br>tri-grams: 0.5047</br>4-grams: 0.3897</br><hr>loss: 0.2378</br>acc: 0.9813</br>val_loss: 2.1694</br>val_acc: 0.7024",
    "Italian" : "1-grams: 0.7932</br>bi-grams: 0.7146</br>tri-grams: 0.6588</br>4-grams: 0.4915</br><hr>loss: 0.1287</br>acc: 0.9840</br>val_loss: 1.0991</br>val_acc: 0.8213",
    "Spanish" : "1-grams: 0.6442</br>bi-grams: 0.5233</br>tri-grams: 0.4715</br>4-grams: 0.3637</br><hr>loss: 0.2208</br>acc: 0.9840</br>val_loss: 2.2772</br>val_acc: 0.7074",
    "Turkish" : "1-grams: 0.6796</br>bi-grams: 0.5705</br>tri-grams: 0.5154</br>4-grams: 0.3732</br><hr>loss: 0.2303</br>acc: 0.9767</br>val_loss: 2.1991</br>val_acc: 0.6970"
}

lang_details = {
    "German" : "German Vocabulary Size: 13,834</br>German Max Sentence Length: 17</br><hr>English Vocabulary Size: 7,910</br>English Max Sentence Length: 8",
    "French" : "French Vocabulary Size: 15,378</br>French Max Sentence Length: 14</br><hr>English Vocabulary Size: 7,468</br>English Max Sentence Length: 8",
    "Italian" : "Italian Vocabulary Size: 11772</br>Italian Max Sentence Length: 17</br><hr>English Vocabulary Size: 5296</br>English Max Sentence Length: 7",
    "Spanish" : "Spanish Vocabulary Size: 16,831</br>Spanish Max Sentence Length: 14</br><hr>English Vocabulary Size: 8,943</br>English Max Sentence Length: 10",
    "Turkish" : "Turkish Vocabulary Size: 23,521</br>Turkish Max Sentence Length: 9</br><hr>English Vocabulary Size: 8,183</br>English Max Sentence Length: 7"

}

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
    # set_language(lang_index)
    return render_template('index.html',
                            translation='',
                            options=lang_options,
                            selected_lang=get_selected(lang_options),
                            lang_details=lang_details[lang_index],
                            bleu_score=bleus[lang_index])

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
                                    options=lang_options,
                                    lang_details=lang_details[lang_index],
                                    bleu_score=bleus[lang_index])

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
                                options=lang_options,
                                lang_details=lang_details[lang_index],
                                bleu_score=bleus[lang_index])

        # for option in options:
        #     option["Selected"] = True if option["Value"] == lang_index else False
        # return render_template('index.html', input_text=input, translation=translation, selected_lang=get_selected(options), options=options)

if __name__ == '__main__':
    app.run(debug=True)
