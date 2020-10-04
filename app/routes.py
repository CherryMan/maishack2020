import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from app import app
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
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(path)
            file.save(path)
            return redirect(url_for('uploaded_file',
                                    filename=filename,
                                    prediction = "chelsea"))
    return render_template('upload.html')
    

@app.route('/catbreedfound', methods=['GET', 'POST'])
def uploaded_file():
    return render_template(
        'display.html',
        filename=f"{app.config['UPLOAD_FOLDER']}\{request.args.get('filename')}",
        prediction=request.args.get('prediction'))
