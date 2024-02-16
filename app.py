from flask import Flask, render_template, request, session
import os
from Resources.lib import *
from Resources import lib as lib
from main import *
import main as m

app = Flask(__name__)
app.secret_key = 'Puneet'
UPLOAD_FOLDER = "Upload"
LOGS = "logs"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

log_file = "./logs/web_app.log"

# logging.basicConfig(filename=log_file, level=logging.INFO)

lg_info = lib.setup_logger("WEB_APP",log_file , level=lib.logging.INFO)

lg_err = lib.setup_logger("WEB_APP",log_file , level=lib.logging.ERROR)

lg_war = lib.setup_logger("WEB_APP",log_file , level=lib.logging.WARNING)

@app.route('/')
def index():
    if os.path.exists('logs/upload_flag.txt') == True:
        os.remove('logs/upload_flag.txt')
    return render_template('index.html')

@app.route('/about-project')
def about_project():
    if os.path.exists('logs/upload_flag.txt') == True:
        os.remove('logs/upload_flag.txt')
    return render_template('about-project.html')

@app.route('/about-team')
def about_team():
    if os.path.exists('logs/upload_flag.txt') == True:
        os.remove('logs/upload_flag.txt')
    return render_template('about-team.html')

@app.route('/upload-page')
def upload():
    if os.path.exists('logs/upload_flag.txt') == True:
        os.remove('logs/upload_flag.txt')
    return render_template('upload-page.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if os.path.exists('logs/upload_flag.txt') == True:
        os.remove('logs/upload_flag.txt')
    with open('logs/upload_flag.txt', 'w') as fp:
        pass
    if 'file' not in request.files:
        lg_err.error('ERROR: No file part')
        return 'Error: No file part'

    file = request.files['file']

    if file.filename == '':
        lg_war.warning('WARNING: No Selected file')
        return 'Error: No selected file'

    try:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        lg_info.info("File Uploaded SUCCESSFULLY")
        session['File_Name'] = filename
        return 'File uploaded SUCCESSFULLY.'
    except Exception as e:
        lg_err.error(f'File uploaded FAILED{e}')
        return f'Error: {str(e)}'
    
@app.route('/result', methods=['POST' , 'GET'])
def result():
    if os.path.exists('logs/upload_flag.txt') == True:
        os.remove('logs/upload_flag.txt')
    else:
        return render_template('result.html',data="Please Upload a File")
    file = session.get('File_Name' , None)
    lg_info.info(f"Classification STARTED of File : {file}")
    if file == None:
        return render_template('result.html',data="Please Upload a File")
    data = m.main(file)
    if data != "":
        lg_info.log(level=4,msg=f"Classification DONE of File : {file}")
        print(file)
        return render_template('result.html',data=data)
    else:
        lg_err.error(f"Classification FAILED : {file}")
        return render_template('result.html',data=data)
    
    
if __name__ == '__main__':
    if not os.path.exists(LOGS):
        os.makedirs(LOGS)
        lg_info.info(f"{LOGS} Created SUCCESSFULLY")
    elif not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        lg_info.info(f"{UPLOAD_FOLDER} Created SUCCESSFULLY")
    lg_info.info(f"APP STARTED SUCCESSFULLY")
    app.run(debug=True , port=5000,host='0.0.0.0')
