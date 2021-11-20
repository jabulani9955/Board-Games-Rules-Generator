from flask import Flask, render_template, request
import random
import textwrap
from model import generate
from text_processing import processing


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('prediction.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        prompt = request.form.get('input').capitalize()
        result = processing(textwrap.fill(generate(prompt), 150))
    return render_template('prediction.html', result=result, prompt=prompt)


if __name__ == '__main__':
    app.run()
