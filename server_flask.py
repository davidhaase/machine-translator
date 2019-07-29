import os
from flask import Flask, render_template, request
from trans import Translator
from keras.backend import clear_session

languages = {'Français': {'s3_file':'LanguageTexts/deu.txt', 'source':'de', 'target':'en', 'model_path':'models/de_to_en/pickles/model_prefs.pkl'},
            'Deutsch': {'s3_file':'LanguageTexts/deu.txt', 'source':'fr', 'target':'en', 'model_path':'models/de_to_en/pickles/model_prefs.pkl'},
            'Italiano': {'s3_file':'LanguageTexts/deu.txt', 'source':'it', 'target':'en', 'model_path':'models/de_to_en/pickles/model_prefs.pkl'},
            'Español': {'s3_file':'LanguageTexts/deu.txt', 'source':'es', 'target':'en', 'model_path':'models/de_to_en/pickles/model_prefs.pkl'}}

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
        model_pref_path = languages[lang_index]['model_path']
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
