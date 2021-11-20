from flask import Flask, render_template, request
import random
import textwrap
from model import generate
from text_processing import processing


app = Flask(__name__, static_folder='templates/')


@app.route('/')
def index():
    return render_template('prediction.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        prompt = request.form.get('input')
        result = processing(textwrap.fill(generate(prompt, len_gen=100, temperature=.8), 120))
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run()
