
class ClassifierModel(nn.Module):
  def __init__(self, model="b0"):
    self.model = EfficientNet.from_pretrained(f"efficientnet-{model}", advprop=True, num_classes=1)

  def forward(self, x):
    return self.model(x)


model = EfficientNet.from_pretrained(f"efficientnet-{model}", advprop=True, num_classes=1)



class Data(Dataset):
  def __init__(self, img_dir):
    self.img_dir = img_dir
    self.training_imgs_names = os.listdir(f"{img_dir}/images/")
    random.shuffle(self.training_imgs_names)

  def __len__(self):
    return len(self.training_imgs_names)

  def __getitem__(self, index):
    img_name = self.training_imgs_names[index]

    img = cv2.imread(self.img_dir + "/images/" + img_name) #input img
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)

    img = img.transpose(2,0,1)

    img_mask = cv2.imread(f"{self.img_dir}/masks/{img_name}", cv2.IMREAD_GRAYSCALE)
    label = 0 if np.mean(img_mask) == 0 else 1
    return img, label

