class Data():
  def __init__(self, img_dir, train_df, img_dim, s_img_dim):
    self.img_dir = img_dir
    self.training_imgs_names = os.listdir(f"{img_dir}/images/")
    random.shuffle(self.training_imgs_names)
    self.df_imgs  = {}

    for i in range(train_df.shape[0]):
      img_id = train_df.iloc[i,0]
      self.df_imgs[img_id] = rasterio.open(f"/kaggle/input/hubmap-kidney-segmentation/train/{img_id}.tiff")

    self.img_dim = img_dim
    self.s_img_dim = s_img_dim
    self.diff_dim = (s_img_dim - img_dim) // 2

  def __len__(self):
    return len(self.training_imgs_names)

  def __getitem__(self, index):
    img_name = self.training_imgs_names[index]

    #### s_img_dim coords
    img_info = re.sub(".jpg", "", img_name).split("_")

    h1 = int(img_info[1]) - self.diff_dim if int(img_info[1]) - self.diff_dim > 0 else 0
    h2 = self.img_dim + self.diff_dim

    w1 = int(img_info[2]) - self.diff_dim if int(img_info[2]) - self.diff_dim > 0 else 0
    w2 = self.img_dim + self.diff_dim

    s_img = self.df_imgs{img_info[0]}.read([1,2,3], window=Window.from_slices((h1,h2), (w1,w2)))
    s_img = np.moveaxis(s_img,0,-1)
    ####

    # making s_img of proper size by adding const padding
    if s_img.shape != (self.s_img_dim, self.s_img_dim, 3):
      if s_img.shape[0] % 2 != 0:
        top = (self.s_img_dim - s_img.shape[0]) // 2 + 1
        bot = (self.s_img_dim - s_img.shape[0]) // 2
      else:
        top = (self.s_img_dim - s_img.shape[0]) // 2
        bot = (self.s_img_dim - s_img.shape[0]) // 2

      if s_img.shape[1] % 2 != 0:
        left = (self.s_img_dim - s_img.shape[1]) // 2 + 1
        right = (self.s_img_dim - s_img.shape[1]) // 2
      else:
        left = (self.s_img_dim - s_img.shape[1]) // 2
        right = (self.s_img_dim - s_img.shape[1]) // 2
      s_img = cv2.copyMakeBorder(s_img, top,bot,left,right,cv2.BORDER_CONSTANT, value=[255,255,255])

    img = cv2.imread(self.img_dir + "/images/" + img_name) #input img
    img_mask = cv2.imread(f"{self.img_dir}/masks/{img_name}", cv2.IMREAD_GRAYSCALE)

    if len(img_info) == 5:
      if img_info[-1] == "aug1":
        s_img = cv2.flip(s_img, 1)
      elif img_info[-1] == "aug2":
        s_img = cv2.flip(s_img, 0)
      elif img_info[-1] == "aug3":
        s_img = cv2.flip(s_img, -1)

    img = img.transpose(2,0,1)
    s_img = s_img.transpose(2,0,1)

    return img, s_img, img_mask

