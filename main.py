from flask import Flask
from flask_session import Session
from flask import request, render_template, session, Markup, url_for, redirect, send_file
import pickle, datetime, os, shutil, random

from PIL import Image
import torch
import torchvision

from models import Model_

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
sess = Session()


def net_forward(path):
    return model(path)


@app.route('/')
def main():
    return redirect(url_for('handle'))


@app.route('/handle', methods=['GET'])
def home():
    path = "https://i.ibb.co/ZVFsg37/default.png"
    return render_template('main.html', path=path)


@app.route('/handle', methods=['POST'])
def handle():
    image = request.files['img_one']
    print(image)
    unic_name = 'static/' + ''.join([str(random.randint(0, 9)) for _ in range(20)]) + '.jpg'
    print(unic_name)
    image.save(unic_name)

    output = 'это просто тест это просто тест'
    print(output)
    output = f'''
        <div style="font-size: 15px; 
            color: white;
	        font-family: cursive;
	        text-align: center;">
            {output}
        </div>
    '''
    return render_template('main.html', path=url_for('static', filename=unic_name[7:]), caption=output)


if __name__ == '__main__':
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)

    #print('init model!')
    #model = Model_()
    print('start app!')

    app.run(debug=True)