from flask import Flask, render_template, request
from translator_new import Translator

app = Flask(__name__)

languages = {'Français': {'s3_file':'LanguageTexts/deu.txt', 'source':'de', 'target':'en'},
            'Deutsch': {'s3_file':'LanguageTexts/deu.txt', 'source':'fr', 'target':'en'},
            'Italiano': {'s3_file':'LanguageTexts/deu.txt', 'source':'it', 'target':'en'},
            'Español': {'s3_file':'LanguageTexts/deu.txt', 'source':'es', 'target':'en'}}

@app.route('/')
def home_screen():
    return render_template('index.html', translation='')


@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':

        result = request.form
        for item in result.items():
            key, value = item
            if key == 'topping':
                lang_index = value
                continue
            if key == 'Input_Text':
                input = value
        print(str(lang_index))
        s3_file = languages[lang_index]['s3_file']
        source = languages[lang_index]['source']
        target = languages[lang_index]['target']
        Lang = Translator(s3_file, source, target)

        prediction, bleu_scores = Lang.translate(input)
        return render_template('index.html', input_text=prediction['input_text'], translation=prediction['translation'], bleu_scores=bleu_scores)

if __name__ == '__main__':
    app.run(debug = True)

# @app.route('/')
# def hello_world():
#     author = "Edgar"
#     name = "David"
#     return render_template('index.html', author=author, name=name)

# def hello_world():
#     return 'hello world'
#
# app.add_url_rule('/', 'hello', hello_world)
