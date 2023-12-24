from flask import Flask, render_template, request, redirect, url_for , session
import os
# from Resources.lib import *
# from Resources import lib as lib
from main import *
import main as m

app = Flask(__name__)
app.secret_key = 'Puneet'
UPLOAD_FOLDER = "upload"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Error: No file part'

    file = request.files['file']

    if file.filename == '':
        return 'Error: No selected file'

    try:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        session['File_Name'] = filename
        return 'File uploaded successfully.'
    except Exception as e:
        return f'Error: {str(e)}'

@app.route('/result', methods=['POST' , 'GET'])
def result():
    file = session.get('File_Name' , None)
    print(file)
    data = m.main(file)
    return render_template('result.html',data=data)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
