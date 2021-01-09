
#feeds data to the NN
class Data():
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
    
    if img_mask is None and x_img is None:
      print("img_mask and x_img is none")
    elif x_img is None:
      print("x_img is none")
    elif img_mask is None:
      print(img_name)
      print("img_mask is None")
    
    return x_img, img_mask

