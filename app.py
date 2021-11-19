from flask import Flask, render_template, request
import random
from model import generate
import textwrap


app = Flask(__name__, static_folder='templates/')


@app.route('/')
def index():
    return render_template('prediction.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        prompt = request.form.get('input')
        result = textwrap.fill(generate(prompt, len_gen=100, temperature=.8), 120).strip().title()
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run()
