
import collections
import json
import os
import uuid
import gc
import random

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import tifffile as tiff
import seaborn as sns
from skimage.measure import label, regionprops

TRAIN_PATH = "../input/hubmap-kidney-segmentation/train"

train_df = pd.read_csv("../input/hubmap-kidney-segmentation/train.csv")
os.system("mkdir training_data")

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def change_brightness(img_arr, min_val=0.8, max_val=1.3):
  """
    Changes the brightness of image.
  """
  value = random.uniform(min_val, max_val)
  hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
  hsv = np.array(hsv, dtype=np.float64)
  hsv[:,:,1] = hsv[:,:,1]*value
  hsv[:,:,1][hsv[:,:,1]>255] = 255
  hsv[:,:,2] = hsv[:,:,2]*value
  hsv[:,:,2][hsv[:,:,2]>255] = 255
  hsv = np.array(hsv, dtype=np.uint8)
  img_arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return img_arr

def get_smaller_imgs(img_id, img, img_mask, small_img_width=224, small_img_height=224, stride=100, prob_rand_img=0.08, prob_blur_img=0.3):
  imgheight, imgwidth, imgchannels = img.shape

  curr_h = 0
  small_img_id = 0
  while curr_h + small_img_height <= imgheight:
    curr_w = 0
    while curr_w + small_img_width <= imgwidth:
      smallimg_arr = img[curr_h:curr_h+small_img_height, curr_w:curr_w+small_img_width, :]
      smallimg_mask_arr = img_mask[curr_h:curr_h+small_img_height, curr_w:curr_w+small_img_width]

      num = random.random()

      if num <= prob_rand_img: #adds random image of kidney without glomeruli
        img_mean = np.mean(smallimg_arr)

        if img_mean < 185.0 and img_mean > 25.0: #checks for completes white and black images

          cv2.imwrite("training_data/images/{}_{}x{}_{}.jpg".format(img_id, curr_h, curr_w, stride), smallimg_arr)
          cv2.imwrite("training_data/masks/{}_{}x{}_{}.jpg".format(img_id, curr_h, curr_w, stride), smallimg_mask_arr)
          small_img_id += 1
      else:
        img_mean = np.mean(smallimg_mask_arr)

        if img_mean >= 0.1: #checks for images with glomeruli

          #img name format img_id,curr_h,curr_w,stride

          cv2.imwrite("training_data/images/{}_{}x{}_{}.jpg".format(img_id, curr_h, curr_w, stride), smallimg_arr)
          cv2.imwrite("training_data/masks/{}_{}x{}_{}.jpg".format(img_id, curr_h, curr_w, stride), smallimg_mask_arr)

          #IMAGE AUGMENTATION

          #image augment #1

          smallimg_arr = cv2.flip(smallimg_arr, 1)
          smallimg_mask_arr = cv2.flip(smallimg_mask_arr, 1)

          cv2.imwrite("training_data/images/{}_{}x{}_{}_aug1.jpg".format(img_id, curr_h, curr_w, stride), smallimg_arr)
          cv2.imwrite("training_data/masks/{}_{}x{}_{}_aug1.jpg".format(img_id, curr_h, curr_w, stride), smallimg_mask_arr)


          #image augment #2

          prob = random.random()

          if prob < prob_blur_img: # add blur to the image with certain probability
            smallimg_arr_2 = cv2.blur(smallimg_arr, (2,2))
            smallimg_arr_2 = cv2.flip(smallimg_arr_2, 0)
            smallimg_mask_arr_2 = cv2.flip(smallimg_mask_arr, 0)
          else:
            smallimg_arr_2 = cv2.flip(smallimg_arr, 0)
            smallimg_mask_arr_2 = cv2.flip(smallimg_mask_arr, 0)

          cv2.imwrite("training_data/images/{}_{}x{}_{}_aug2.jpg".format(img_id, curr_h, curr_w, stride), smallimg_arr_2)
          cv2.imwrite("training_data/masks/{}_{}x{}_{}_aug2.jpg".format(img_id, curr_h, curr_w, stride), smallimg_mask_arr_2)

          #image augment #3

          smallimg_arr = change_brightness(smallimg_arr)

          cv2.imwrite("training_data/images/{}_{}x{}_{}_aug3.jpg".format(img_id, curr_h, curr_w, stride), smallimg_arr)
          cv2.imwrite("training_data/masks/{}_{}x{}_{}_aug3.jpg".format(img_id, curr_h, curr_w, stride), smallimg_mask_arr)

          small_img_id += 4

      curr_w += stride
    curr_h += stride
  print("Total {} images were written.".format(small_img_id))


#### creates dataset ####

os.system(f"mkdir training_data/images")
os.system(f"mkdir training_data/masks")

for i in range(train_df.shape[0]):
  img_id = train_df.iloc[i, 0]
  img = tiff.imread(f"{TRAIN_PATH}/{img_id}.tiff")
  if len(img.shape) == 5:
    img = img[0][0].transpose(1, 2, 0)

  print(f"completed reading image, image id : {img_id}, index : {i}, img shape : {img.shape}")

  img_mask = rle2mask(train_df.iloc[i, 1], shape=(img.shape[1], img.shape[0]))
  print(f"The shape of the mask is {img_mask.shape}")

  print("writing smaller imgs")

  get_smaller_imgs(img_id, img, img_mask)

  print("completed")
  print()

