def mask2rle(img):
  pixels= img.T.flatten()
  pixels = np.concatenate([[0], pixels, [0]])
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape):
  s = mask_rle.split()
  starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
  starts -= 1
  ends = starts + lengths
  img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
  for lo, hi in zip(starts, ends):
    img[lo:hi] = 1
  return img.reshape(shape).T

def change_brightness(img_arr, min_val=0.9, max_val=1.1):
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

def get_smaller_imgs(img_id, img, img_mask, img_dim, s_img_dim, stride=100, prob_rand_img=0.08, prob_blur_img=0.3, type="segmentation"):
  imgheight, imgwidth, imgchannels = img.shape

  curr_h = 0
  small_img_id = 0
  non_glomeruli_img = 0
  glomeruli_img = 0

  diff_dim = (s_img_dim - img_dim) // 2

  while curr_h + img_dim <= imgheight:
    curr_w = 0
    while curr_w + img_dim <= imgwidth:
      smallimg_arr = img[curr_h:curr_h+img_dim, curr_w:curr_w+img_dim, :]
      smallimg_mask_arr = img_mask[curr_h:curr_h+img_dim, curr_w:curr_w+img_dim]

      #### Coords for s_img
      h1 = curr_h - diff_dim if curr_h - diff_dim > 0 else 0
      h2 = curr_h + img_dim + diff_dim

      w1 = curr_w - diff_dim if curr_w - diff_dim > 0 else 0
      w2 = curr_w + img_dim + diff_dim

      s_img = img[h1:h2, w1:w2, :]

      if s_img.shape != (s_img_dim, s_img_dim, 3):
        if s_img.shape[0] % 2 != 0:
          top = (s_img_dim - s_img.shape[0]) // 2 + 1
          bot = (s_img_dim - s_img.shape[0]) // 2
        else:
          top = (s_img_dim - s_img.shape[0]) // 2
          bot = (s_img_dim - s_img.shape[0]) // 2

        if s_img.shape[1] % 2 != 0:
          left = (s_img_dim - s_img.shape[1]) // 2 + 1
          right = (s_img_dim - s_img.shape[1]) // 2
        else:
          left = (s_img_dim - s_img.shape[1]) // 2
          right = (s_img_dim - s_img.shape[1]) // 2
        s_img = cv2.copyMakeBorder(s_img, top,bot,left,right,cv2.BORDER_CONSTANT, value=[255,255,255])
      ####
      smallimg_arr = cv2.resize(smallimg_arr, (smallimg_arr.shape[0]//2, smallimg_arr.shape[1]//2), interpolation=cv2.INTER_AREA)
      smallimg_mask_arr = cv2.resize(smallimg_mask_arr, (smallimg_mask_arr.shape[0]//2, smallimg_mask_arr.shape[1]//2), interpolation=cv2.INTER_AREA)
      s_img = cv2.resize(s_img, (s_img.shape[0]//2, s_img.shape[1]//2), interpolation=cv2.INTER_AREA)


      num = random.random()
      if num <= prob_rand_img: #adds random image of kidney without glomeruli
        img_mean = np.mean(smallimg_arr)
        img_mask_mean = np.mean(smallimg_mask_arr)

        cv2.imwrite("/kaggle/temp/training_data/{}/images/{}_{}_{}_{}.jpg".format(\
                                                                    type, img_id, curr_h, curr_w, stride), smallimg_arr)
        cv2.imwrite("/kaggle/temp/training_data/{}/s_images/{}_{}_{}_{}.jpg".format(\
                                                                    type, img_id, curr_h, curr_w, stride), s_img)
        cv2.imwrite("/kaggle/temp/training_data/{}/masks/{}_{}_{}_{}.jpg".format(\
                                                                    type, img_id, curr_h, curr_w, stride), smallimg_mask_arr)
        small_img_id += 1
        non_glomeruli_img += 1

      else:
        img_mean = np.mean(smallimg_mask_arr)

        if img_mean >= 0.05: #checks for images with glomeruli

          #img name format img_id,curr_h,curr_w,stride

          cv2.imwrite("/kaggle/temp/training_data/{}/images/{}_{}_{}_{}_aug0.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_arr)
          cv2.imwrite("/kaggle/temp/training_data/{}/s_images/{}_{}_{}_{}_aug0.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), s_img)
          cv2.imwrite("/kaggle/temp/training_data/{}/masks/{}_{}_{}_{}_aug0.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_mask_arr)
          #IMAGE AUGMENTATION

          #image augment #1

          smallimg_arr_1 = cv2.flip(smallimg_arr, 1)
          s_img_1 = cv2.flip(s_img, 1)
          smallimg_mask_arr_1 = cv2.flip(smallimg_mask_arr, 1)

          cv2.imwrite("/kaggle/temp/training_data/{}/images/{}_{}_{}_{}_aug1.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_arr_1)
          cv2.imwrite("/kaggle/temp/training_data/{}/s_images/{}_{}_{}_{}_aug1.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), s_img_1)
          cv2.imwrite("/kaggle/temp/training_data/{}/masks/{}_{}_{}_{}_aug1.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_mask_arr_1)
          #image augment #2

          prob = random.random()

          if prob < prob_blur_img: # add blur to the image with certain probability
            smallimg_arr_2 = cv2.blur(smallimg_arr, (2,2))
            smallimg_arr_2 = cv2.flip(smallimg_arr_2, 0)
            s_img_2 = cv2.blur(s_img, (2,2))
            s_img_2 = cv2.flip(s_img_2, 0)
            smallimg_mask_arr_2 = cv2.flip(smallimg_mask_arr, 0)
          else:
            smallimg_arr_2 = cv2.flip(smallimg_arr, 0)
            s_img_2 = cv2.flip(s_img, 0)
            smallimg_mask_arr_2 = cv2.flip(smallimg_mask_arr, 0)

          cv2.imwrite("/kaggle/temp/training_data/{}/images/{}_{}_{}_{}_aug2.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_arr_2)
          cv2.imwrite("/kaggle/temp/training_data/{}/s_images/{}_{}_{}_{}_aug2.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), s_img_2)
          cv2.imwrite("/kaggle/temp/training_data/{}/masks/{}_{}_{}_{}_aug2.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_mask_arr_2)

          #image augment #3

          smallimg_arr_3 = change_brightness(smallimg_arr)
          s_img_3 = change_brightness(s_img)

          smallimg_arr_3 = cv2.flip(smallimg_arr_3, -1)
          s_img_3 = cv2.flip(s_img_3, -1)
          smallimg_mask_arr_3 = cv2.flip(smallimg_mask_arr, -1)

          cv2.imwrite("/kaggle/temp/training_data/{}/images/{}_{}_{}_{}_aug3.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_arr_3)
          cv2.imwrite("/kaggle/temp/training_data/{}/s_images/{}_{}_{}_{}_aug3.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), s_img_3)
          cv2.imwrite("/kaggle/temp/training_data/{}/masks/{}_{}_{}_{}_aug3.jpg".format(\
                                                                            type, img_id, curr_h, curr_w, stride), smallimg_mask_arr_3)

          small_img_id += 4
          glomeruli_img += 4

      curr_w += stride
    curr_h += stride
  print("Total {} images were written. of which {} were non glomeruli img and {} were glomeruli img".format(\
                                                                            small_img_id, non_glomeruli_img, glomeruli_img))


%%time
os.system(f"mkdir /kaggle/temp/training_data/segmentation")
os.system(f"mkdir /kaggle/temp/training_data/segmentation/images")
os.system(f"mkdir /kaggle/temp/training_data/segmentation/s_images")
os.system(f"mkdir /kaggle/temp/training_data/segmentation/masks")

#### creates dataset for classification and segmentation ####

for i in range(train_df.shape[0]):
  if i == 1:
    continue
  img_id = train_df.iloc[i, 0]
  img = tiff.imread(f"{TRAIN_PATH}/{img_id}.tiff")
  if len(img.shape) == 5:
    img = img[0][0].transpose(1, 2, 0)

  print(f"completed reading image, image id : {img_id}, index : {i}, img shape : {img.shape}")

  img_mask = rle2mask(train_df.iloc[i, 1], shape=(img.shape[1], img.shape[0]))
  print(f"The shape of the mask is {img_mask.shape}")

  print("writing smaller imgs for segmentation")
  get_smaller_imgs(img_id, img, img_mask, img_dim=IMG_DIM, s_img_dim=S_IMG_DIM, stride=85, prob_rand_img=0.08, type="segmentation")

  del img, img_mask
  gc.collect()
  print("completed")
  print()


