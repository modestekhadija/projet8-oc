from flask import Flask, url_for, redirect, render_template, request, flash
from werkzeug.utils import secure_filename
from . import models
import requests
import os



app = Flask(__name__)
app.config.from_object('config')
app.config['SECRET_KEY']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER_IMG'], filename))
            return redirect(url_for('show_image', filename=filename))

    return render_template("upload.html")

@app.route('/index/<filename>',methods=['GET', 'POST'])
def show_image(filename):

    url =  url_for('static', filename= 'tmp/img/'  + filename)
    
    return render_template("index.html", url=url, filename=filename)

@app.route('/result/<filename>',methods=['GET'])
def show_mask(filename):

    urlapi = 'https://predict-mask.azurewebsites.net/predict'
    path =  os.path.join(app.config['UPLOAD_FOLDER_IMG'], filename)
    files = {'img': open(path, 'rb')}
    r = requests.post(urlapi, files=files) 
    
    mask_path =  os.path.join(app.config['UPLOAD_FOLDER_MASK'], filename)

    if r.status_code == 200:
        with open(mask_path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
    url1 = url_for('static', filename= 'tmp/img/'  + filename)
    url2 = url_for('static', filename= 'tmp/mask/'  + filename)
    real_mask_filename = models.mask_path_fct(filename)
    ville = filename.split('_')[0]
    real_mask_path =  os.path.join(app.config['MASKS_FOLDER'], ville + "/" + real_mask_filename)

    try : 
        real_mask_colored = models.transform_real_mask_to_colored_mask(real_mask_path)

        real_mask_colored.save(os.path.join(app.config['UPLOAD_FOLDER_MASK'], real_mask_filename))
    
        url3 = url_for('static', filename =  'tmp/mask/'  + real_mask_filename)
        print(url3)


    except:

        message = "Le masque réel n'existe pas dans la base de données"
        return render_template("result2.html", url1=url1 , url2=url2 , message=message)


    return render_template("result.html", url1=url1 , url2=url2 , url3=url3)
    
