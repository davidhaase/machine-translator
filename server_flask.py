import os
from flask import Flask, render_template, request
from translator import Translator
from keras.backend import clear_session


app = Flask(__name__)
model_id = 'basic_50K/'
selected_lang = 'Deutsch'
lang_prefix = {'Français':'fr_to_en/',
                'Deutsch':'de_to_en/',
                'Italiano':'it_to_en/',
                'Español':'es_to_en/'}

@app.route('/')
def home_screen():
    return render_template('index.html', translation='')


@app.route('/set_language')
def set_language():
    if request.method == 'POST':

        # Get the results from the web user
        result = request.form
        for item in result.items():
            print(item)
    return render_template('index.html', translation='', selected_lang='German')


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
        model_pref_path = 'models/' + lang_prefix[selected_lang] + model_id + 'pickles/model_prefs.pkl'
        if not os.path.isfile(model_pref_path):
            input == 'Error: ' + input
            translation = 'No Model found for {}'.format(lang_index)
            return render_template('index.html', input_text=input, translation=translation)

        # A model exists, so use it and translate away!
        print(model_pref_path)
        T = Translator(model_pref_path)
        translation = T.translate(input)

        # Keras backend needs to clear the session
        clear_session()
        return render_template('index.html', input_text=input, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)
