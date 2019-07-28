from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home_screen():
   return render_template('index.html', translation='')


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template('index.html', translation = get_translation(result))

def get_translation(input_text):
    translated = 'bonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\nbonjour, le monde!\n'
    output = 'You entered: ' + str(input_text.items()) + '\n' + translated

    return output

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
