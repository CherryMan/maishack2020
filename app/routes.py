import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from app import app
from model import model_eval
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = f"{app.config['UPLOAD_FOLDER']}/{filename}"
            file.save(path)
            return redirect(url_for('prediction', filename=filename))
    return render_template('upload.html')
    

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    filename = request.args.get('filename')
    pred, probs = model_eval(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    probs = sorted(list(probs.items()), key=lambda x: x[1], reverse=True)

    return render_template(
        'display.html',
        filename=filename,
        prediction=pred,
        probabilities=probs)
