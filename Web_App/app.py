from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processing', methods=['POST'])
def processing():
    if 'audio_file' not in request.files:
        return "No file part"

    file = request.files['audio_file']

    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input_audio.wav')
        file.save(filename)
        return render_template('processing.html')

@app.route('/results')
def results():
    # Perform your audio processing here and prepare data for results
    # For now, let's just display a dummy result
    result_text = "This is a dummy result text."
    result_image = "/static/dummy_image.jpg"
    return render_template('results.html', result_text=result_text, result_image=result_image)

if __name__ == '__main__':
    app.run(debug=True)
