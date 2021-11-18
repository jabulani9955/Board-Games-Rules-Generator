from flask import Flask, render_template, request
import random
from model import generate
import textwrap

app = Flask(__name__)


# @app.route('/home')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    return render_template('prediction.html')



@app.route('/results', methods=['POST'])
def predict():
    if request.method == 'POST':
        prompt = request.form.get('input')
        result = textwrap(generate(prompt, len_gen=100, temperature=.8))
    return render_template('prediction.html', result=result)


if __name__ == '__main__':
    app.run()
