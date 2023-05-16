from flask import Flask
from os.path import join, dirname, realpath

UPLOAD_FOLDER_IMG = join(dirname(realpath(__file__)), 'flaskapp/static/tmp/img')

UPLOAD_FOLDER_MASK = join(dirname(realpath(__file__)), 'flaskapp/static/tmp/mask')

IMAGES_FOLDER = join(dirname(realpath(__file__)), 'flaskapp/static/images/')
MASKS_FOLDER = join(dirname(realpath(__file__)), 'flaskapp/static/masks/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SECRET_KEY = "12345654213"