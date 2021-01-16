
class Data():
  def __init__(self, img_dir, img_height, img_width, patch_size=8, img_features=IMG_FEATURES):
    self.img_dir = img_dir
    self.img_height = img_height
    self.img_width = img_width
    self.img_features = img_features
    self.patch_size = patch_size

    self.training_imgs_names = os.listdir(f"{img_dir}/images/")
    random.shuffle(self.training_imgs_names)

  @njit()
  def get_image_patches(self, img, patch_size, img_features):
    idx = 0
    num_images = (img.shape[0]//patch_size)**2
    patches = np.zeros((num_images, patch_size, patch_size, img_features), dtype=np.uint8)
    for i in range(0, img.shape[0], patch_size):
      for j in range(0, img.shape[0], patch_size):
        patches[idx, :, :, :] = img[i:i+patch_size, j:j+patch_size]
        idx += 1
    return patches
      

  def __len__(self):
    return len(self.training_imgs_names)

  def __getitem__(self, index):
    img_name = self.training_imgs_names[index]
    x_img = cv2.imread(self.img_dir + "/images/" + img_name)
    x_img = get_image_patches(x_img, self.patch_size, self.img_features)

    img_mask = cv2.imread(f"{self.img_dir}/masks/{img_name}", cv2.IMREAD_GRAYSCALE)
    img_mask = get_image_patches(np.expand_dims(img_mask, -1), self.patch_size, 1)
    return x_img, np.squeeze(img_mask)

