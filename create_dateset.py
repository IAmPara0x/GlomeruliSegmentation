
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


def get_smaller_imgs(img_id, img, img_mask, small_img_width=224, small_img_height=224, stride=100, prob_rand_img=0.05, prob_blur_img=0.3):
  random_rotate = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]


  imgheight, imgwidth, imgchannels = img.shape

  curr_h = 0
  small_img_id = 0
  while curr_h + small_img_height <= imgheight:
    curr_w = 0
    while curr_w + small_img_width <= imgwidth:
      smallimg = img[curr_h:curr_h+small_img_height, curr_w:curr_w+small_img_width, :]
      smallimg_mask = img_mask[curr_h:curr_h+small_img_height, curr_w:curr_w+small_img_width]

      num = random.random()

      if num <= prob_rand_img: #adds random image of kidney without glomeruli
        img_mean = np.mean(smallimg)

        if img_mean < 185.0 and img_mean > 25.0: #checks for completes white and black images

          smallimg = Image.fromarray(smallimg)
          smallimg_mask = Image.fromarray(smallimg_mask)
          smallimg.save(f"training_data/images/{img_id}_{curr_h}x{curr_w}_{stride}.jpg")
          smallimg_mask.save(f"training_data/masks/{img_id}_{curr_h}x{curr_w}_{stride}.jpg")
          small_img_id += 1
      else:
        img_mean = np.mean(smallimg_mask)

        if img_mean >= 0.1: #checks for images with glomeruli

          #img name format img_id,curr_h,curr_w,stride
          smallimg = Image.fromarray(smallimg)
          smallimg_mask = Image.fromarray(smallimg_mask)

          smallimg.save(f"training_data/images/{img_id}_{curr_h}x{curr_w}_{stride}.jpg")
          smallimg_mask.save(f"training_data/masks/{img_id}_{curr_h}x{curr_w}_{stride}.jpg")

          #image augmentation

          smallimg = ImageOps.mirror(smallimg)
          smallimg_mask = ImageOps.mirror(smallimg_mask)

          smallimg.save(f"training_data/images/{img_id}_{curr_h}x{curr_w}_{stride}_aug1.jpg")
          smallimg_mask.save(f"training_data/masks/{img_id}_{curr_h}x{curr_w}_{stride}_aug1.jpg")

          prob = random.random()

          if prob < prob_blur_img:
            smallimg = smallimg.filter(ImageFilter.BLUR)
            smallimg = ImageOps.flip(smallimg)
            smallimg_mask = ImageOps.flip(smallimg_mask)
          else:
            smallimg = ImageOps.flip(smallimg)
            smallimg_mask = ImageOps.flip(smallimg_mask)

          smallimg.save(f"training_data/images/{img_id}_{curr_h}x{curr_w}_{stride}_aug2.jpg")
          smallimg_mask.save(f"training_data/masks/{img_id}_{curr_h}x{curr_w}_{stride}_aug2.jpg")

          small_img_id += 3

      curr_w += stride
    curr_h += stride
  print(f"Total {small_img_id} images were written.")


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

