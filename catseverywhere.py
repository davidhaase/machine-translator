from flask import Flask, render_template, request
from translator_new import Translator

app = Flask(__name__)

@app.route('/')
def home_screen():
    return render_template('index.html', translation='')


@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        for item in result.items():
            key, value = item
        prediction, bleu_scores = get_translation(value)
        return render_template('index.html', input_text=prediction['input_text'], translation=prediction['translation'], bleu_scores=bleu_scores)

def get_translation(input):
    German = Translator()
    return German.translate(input)

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
