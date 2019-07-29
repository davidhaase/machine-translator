import os
from flask import Flask, render_template, request
from translator import Translator
from keras.backend import clear_session

lang_sources = {'Français': {'s3_file':'LanguageTexts/fra.txt', 'prefix': 'fr_to_en', 'path':'models/fr_to_en/', 'model_pref_path':'models/fr_to_en/pickles/model_prefs.pkl'},
                'Deutsch': {'s3_file':'LanguageTexts/deu.txt', 'prefix': 'de_to_en', 'path':'models/de_to_en/', 'model_pref_path':'models/de_to_en/pickles/model_prefs.pkl'},
                'Italiano': {'s3_file':'LanguageTexts/ita.txt', 'prefix': 'it_to_en','path':'models/it_to_en/', 'model_pref_path':'models/it_to_en/pickles/model_prefs.pkl'},
                'Español': {'s3_file':'LanguageTexts/esp.txt', 'prefix': 'es_to_en','path':'models/es_to_en/', 'model_pref_path':'models/es_to_en/pickles/model_prefs.pkl'}}


app = Flask(__name__)

@app.route('/')
def home_screen():
    return render_template('index.html', translation='')


@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':

        # Get the results from the web user
        result = request.form
        for item in result.items():
            key, value = item
            if key == 'Language':
                lang_index = value
                continue
            if key == 'Input_Text':
                input = value
                continue

        # Get the loaded model
        model_pref_path = lang_sources[lang_index]['model_pref_path']
        if not os.path.isfile(model_pref_path):
            input == 'Error: ' + input
            translation = 'No Model found for {}'.format(lang_index)
            return render_template('index.html', input_text=input, translation=translation)

        # A model exists, so use it and translate away!
        T = Translator(model_pref_path)
        translation = T.translate(input)

        # Keras backend needs to clear the session
        clear_session()
        return render_template('index.html', input_text=input, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)
