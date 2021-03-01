
BATCH_SIZE = 16
IMG_DIM = 512
DOWNSCALE_X = 2

def read_cortex_medulla(file):
    cortex_polys = []
    medulla_polys = []
    with open(file) as jsonfile:
        data = json.load(jsonfile)
        for index in range(data.__len__()):
            if (data[index]['properties']['classification']['name'] == 'Cortex'):
                geom = np.array(data[index]['geometry']['coordinates'])
                cortex_polys.append(geom)
            if (data[index]['properties']['classification']['name'] == 'Medulla'):
                geom = np.array(data[index]['geometry']['coordinates'])
                medulla_polys.append(geom)
    return cortex_polys, medulla_polys

def make_grid(shape, window, min_overlap):
    y, x = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx*ny,4)

def rle2mask(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# changes the brightness of img
def change_brightness(img_arr, min_val=0.8, max_val=1.3):
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

#for augmenting images
def transform_img(img, img_mask, brightness_aug=True):
  # augment 1
  img1 = cv2.flip(img, 1)
  img1_mask = cv2.flip(img_mask, 1)

  # augment 2
  img2 = cv2.flip(img, 0)
  img2_mask = cv2.flip(img_mask, 0)

  # augment 3
  img3 = cv2.flip(img, -1)
  img3_mask = cv2.flip(img_mask, -1)

  #augment 4
  if brightness_aug:
    img4 = change_brightness(img, min_val=0.9, max_val=1.1)
  else:
    img4 = cv2.blur(img, (3,3))

  return (img, img1, img2, img3, img4), (img_mask, img1_mask, img2_mask, img3_mask, img_mask)

def get_ana_val(ana_mask):
  return np.zeros(3)

# Generates images for Neural Net
def get_data_images(idx, img_id, img_dim, downscale, batch_size, device):
  img = rasterio.open(TRAIN_PATH + img_id + ".tiff")
  mask = rle2mask(train_df.iloc[idx, 1], shape=(img.shape[1], img.shape[0]))

  anatomical_mask = np.zeros((img.shape[0], img.shape[1], 3))
  anatomical_file = TRAIN_PATH + f"/{img_id}" + "-anatomical-structure.json"
  cortex_polys, medulla_polys = read_cortex_medulla(anatomical_file)

  # FIXME: comsumes lot of memory.

  # for medulla_poly in medulla_polys:
  #   cv2.fillPoly(anatomical_mask, pts=medulla_poly,color=(255,255,255))

  # for cortex_poly in cortex_polys:
  #   if len(cortex_poly) > 1:
  #     for cortex_pts in cortex_poly:
  #       cv2.fillPoly(anatomical_mask, pts=np.expand_dims(np.array(cortex_pts[0]).astype(np.int32),
  #                                             axis=0),color=(255,0,0))

  anatomical_mask /= 255.0
  img_slices = make_grid(img.shape, window=img_dim, min_overlap=MIN_OVERLAP)

  buffer_img = []
  buffer_img_mask = []
  buffer_img_ana_info = []

  for (x1,x2,y1,y2) in img_slices:

    img_slice = img.read([1,2,3], window=Window.from_slices((x1, x2), (y1, y2)))
    img_slice = np.moveaxis(img_slice, 0, -1)
    img_slice_mask = mask[x1:x2, y1:y2]

    ## Downscaling the image and it's mask
    img_slice = cv2.resize(img_slice, (img_dim//downscale, img_dim//downscale), interpolation=cv2.INTER_AREA)
    img_slice_mask = cv2.resize(img_slice_mask, (img_dim//downscale, img_dim//downscale), interpolation=cv2.INTER_AREA)

    img_slice_ana_mask = anatomical_mask[x1:x2, y1:y2]

    ohe_ana_val = get_ana_val(img_slice_ana_mask) # TODO: write get_ana_val function

    if np.mean(img_slice_mask) >= 0.05:
      if np.random.random() > 0.5:
        aug_imgs, aug_imgs_mask = transform_img(img_slice, img_slice_mask)
      else:
        aug_imgs, aug_imgs_mask = transform_img(img_slice, img_slice_mask, False)

      buffer_img.extend([aug_img.transpose(2,0,1) for aug_img in aug_imgs])
      buffer_img_mask.extend([aug_img_mask for aug_img_mask in aug_imgs_mask])
      buffer_img_ana_info.extend([ohe_ana_val for _ in range(len(aug_imgs))])

    else:
      buffer_img.append(img_slice.transpose(2,0,1))
      buffer_img_mask.append(img_slice_mask)
      buffer_img_ana_info.append(ohe_ana_val)

    if len(buffer_img) >= batch_size:
      batch_img = torch.FloatTensor(buffer_img[:batch_size]).to(device)
      batch_img_mask = torch.tensor(buffer_img_mask[:batch_size]).to(device)
      batch_img_ana_info = torch.FloatTensor(buffer_img_ana_info[:batch_size]).to(device)

      buffer_img = buffer_img[batch_size:]
      buffer_img_mask = buffer_img_mask[batch_size:]
      buffer_img_ana_info = buffer_img_ana_info[batch_size:]

      yield (batch_img, batch_img_ana_info, batch_img_mask)


