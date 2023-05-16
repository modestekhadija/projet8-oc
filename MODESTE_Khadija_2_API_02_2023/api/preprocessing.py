from io import BytesIO
import numpy as np
from matplotlib import colors
from PIL import Image
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import warnings

warnings.filterwarnings('ignore')


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def dice_coeff(y_true, y_pred):
    score = 1 - dice_loss(y_true, y_pred)
    return score

def IoU(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - intersection

    return intersection / denominator


custom_objects = {"dice_coeff" : dice_coeff, 
                  "IoU" : IoU, 
                  "dice_loss" : dice_loss}

model = tf.keras.models.load_model("model/" + "AUG_VGG_UNET_DL", 
                                   custom_objects = custom_objects)
 


ss_categories = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

categories  = {'void': [0],
 'flat': [1],
 'construction': [2],
 'object': [3],
 'nature': [4],
 'sky': [5],
 'human': [6],
 'vehicle': [7]}


def read_image(path, grayscale=False):
    image = load_img(path, target_size = (128,256), grayscale = grayscale)
    return image


def predict_mask_2Darray(image_loaded):

  # preprocessing   
  im_array = img_to_array(image_loaded)/255.
  im_array.resize(1,128,256,3)
  
  # predict 
  prediction = model.predict(im_array, verbose=0)

  mask = np.argmax(prediction, axis=-1) 

  mask = mask.squeeze()
  return mask


def transform_2Darray_mask(mask_array, cats=ss_categories):

  mask_transformed = np.zeros((mask_array.shape[0], mask_array.shape[1],8), dtype='uint8')

  for k in range(8):
    etiq_cat = cats[list(cats.keys())[k]]
        
    for j in etiq_cat:
      mask_transformed[:,:,k] = mask_transformed[:,:,k] | (mask_array==j) 
            
  return mask_transformed 


def transform_mask_to_colored_mask(mask, cats=categories, colors_palette=['#000000', '#804080', '#464646', '#98fb98', '#6b8e23', '#4682b4', '#dc143c', '#00008e']):

    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3))

    for cat in range(len(cats)):
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]

    return tf.keras.preprocessing.image.array_to_img(img_seg)




def predict_img_to_color_mask(img):

  #  lire l'image avec tensorflow
  mask_2Darray = predict_mask_2Darray(img)

  mask = transform_2Darray_mask(mask_2Darray, cats=categories)

  colored_mask = transform_mask_to_colored_mask(mask)

  return colored_mask



def transform_real_mask_to_colored_mask(path):

  #  lire l'image avec tensorflow
  mask_2Darray = img_to_array(load_img(path, target_size = (128,256), grayscale = True))

  mask_2Darray = np.squeeze(mask_2Darray)

  mask = transform_2Darray_mask(mask_2Darray)

  colored_mask = transform_mask_to_colored_mask(mask)

  return colored_mask
