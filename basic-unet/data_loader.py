
#feeds classification data to the NN
class DataClassification():
  def __init__(self, img_dir, img_height, img_width):
    self.img_dir = img_dir
    self.img_height = img_height
    self.img_width = img_width

    self.training_imgs_names = os.listdir(f"{img_dir}/images/")
    random.shuffle(self.training_imgs_names)

  def __len__(self):
    return len(self.training_imgs_names)

  def __getitem__(self, index):
    img_name = self.training_imgs_names[index]
    x_img = cv2.imread(f"{self.img_dir}/images/{img_name}")
    x_img = x_img.reshape(3, self.img_height, self.img_width)

    img_mask = cv2.imread(f"{self.img_dir}/masks/{img_name}", cv2.IMREAD_GRAYSCALE)
    if np.mean(img_mask) == 0:
      label = 0
    else:
      label = 1
    return x_img, label

#feeds segmentation data to the NN
class DataSegmentation():
  def __init__(self, img_dir, img_height, img_width):
    self.img_dir = img_dir
    self.img_height = img_height
    self.img_width = img_width

    self.training_imgs_names = os.listdir(f"{img_dir}/images/")
    random.shuffle(self.training_imgs_names)

  def __len__(self):
    return len(self.training_imgs_names)

  def __getitem__(self, index):
    img_name = self.training_imgs_names[index]
    x_img = cv2.imread(self.img_dir+ "/images/" + img_name)
    x_img = x_img.reshape(3, self.img_height, self.img_width)

    img_mask = cv2.imread(f"{self.img_dir}/masks/{img_name}", cv2.IMREAD_GRAYSCALE)
    return x_img, img_mask, img_name

