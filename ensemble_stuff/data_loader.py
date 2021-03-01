

class Data(Dataset):
  def __init__(self, img_dir, req_classification_imgs:bool, req_s_imgs:bool):
    self.img_dir = img_dir
    self.training_imgs_names = os.listdir(f"{img_dir}/images/")
    random.shuffle(self.training_imgs_names)

    self.req_s_imgs = req_s_imgs
    self.req_classification_imgs = req_classification_imgs

    if self.req_classification_imgs: self.normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)

  def __len__(self):
    return len(self.training_imgs_names)

  def __getitem__(self, index):
    img_name = self.training_imgs_names[index]

    img = cv2.imread(self.img_dir + "/images/" + img_name) #input img
    imgs = np.array([np.rot90(img,i).transpose(2,0,1) for i in range(0,4)])

    if self.req_s_imgs:
      s_img = cv2.imread(self.img_dir + "/s_images/" + img_name) #input img
      s_imgs = np.array([np.rot90(s_img,i).transpose(2,0,1) for i in range(0,4)])
      return imgs, s_imgs, img_name

    return imgs, img_name


