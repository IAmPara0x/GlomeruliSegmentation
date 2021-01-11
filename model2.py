
import torch
import torch.nn as nn
import torch.optim as optim


class ConvBlock(nn.Module):
  def __init__(self, input_dim, in_num_features, kernel_size, dilation=1, stride=1,
      add_dilated_conv=False, is_input_layer=False):
    super(ConvBlock, self).__init__()

    if is_input_layer:
      self.out_num_features = 32
    else:
      self.out_num_features = in_num_features * 2

    conv_padding = calc_equal_conv_padding(input_dim, kernel_size, dilation=1, stride=stride)
    self.conv1 = nn.Conv2d(in_num_features, self.out_num_features, kernel_size=kernel_size, padding=conv_padding)
    self.conv2 = nn.Conv2d(self.out_num_features, self.out_num_features, kernel_size=kernel_size,
                          padding=conv_padding)

    if add_dilated_conv:
      dilated_conv_padding = calc_equal_conv_padding(input_dim, kernel_size, dilation, stride)
      self.conv3 = nn.Conv2d(self.out_num_features, self.out_num_features, kernel_size=kernel_size,
                          dilation=dilation, padding=dilated_conv_padding)
    else:
      self.conv3 = nn.Conv2d(self.out_num_features, self.out_num_features, kernel_size=kernel_size,
                          padding=conv_padding)

    self.output_dim = input_dim // 2

    self.relu = nn.ReLU()
    self.layernorm = nn.LayerNorm(self.out_num_features, self.output_dim, self.output_dim)
    self.maxpool = nn.MaxPool2d(2)

  def forward(self, imgs, impl_residual_conn=True):
    
    imgs = self.conv1(imgs)
    imgs = self.relu(imgs)
    if impl_residual_conn:
      x = imgs
    imgs = self.conv2(imgs)
    imgs = self.relu(imgs)
    imgs = self.conv3(imgs)
    if impl_residual_conn:
      imgs += x
    imgs = self.relu(imgs)
    imgs = self.maxpool(imgs)
    return imgs

  @property
  def out_features(self):
    return self.out_num_features

  @property
  def out_dim(self):
    return self.output_dim


class DetectionModel(nn.Module):
  """
    Model for detection of glomeruli.
  """
  def __init__(self, img_height=224, img_width=224, img_channels=3, layers=5, batch_size=BATCH_SIZE):
    super(DetectionModel, self).__init__()

    self.layers = layers
    self.layers_dict = {}
    input_dim = img_height
    in_num_features = img_channels

    for layer in range(self.layers):
      if layer == 0:
        self.layers_dict[f"convBlock{layer}"] = ConvBlock(input_dim, in_num_features, kernel_size=3,
                                                    dilation=3, stride=1, add_dilated_conv=True, is_input_layer=True)
        input_dim = self.layers_dict[f"convBlock{layer}"].out_dim
        in_num_features = self.layers_dict[f"convBlock{layer}"].out_features
      elif layer == 1:
        self.layers_dict[f"convBlock{layer}"] = ConvBlock(input_dim, in_num_features, kernel_size=3,
                                                    dilation=3, stride=1, add_dilated_conv=True)
        input_dim = self.layers_dict[f"convBlock{layer}"].out_dim
        in_num_features = self.layers_dict[f"convBlock{layer}"].out_features
      elif layer < 4:
        self.layers_dict[f"convBlock{layer}"] = ConvBlock(input_dim, in_num_features, kernel_size=3,
                                                    dilation=2, stride=1, add_dilated_conv=True)
        input_dim = self.layers_dict[f"convBlock{layer}"].out_dim
        in_num_features = self.layers_dict[f"convBlock{layer}"].out_features
      else:
        self.layers_dict[f"convBlock{layer}"] = ConvBlock(input_dim, in_num_features, kernel_size=3,
                                                    stride=1, add_dilated_conv=False)
        input_dim = self.layers_dict[f"convBlock{layer}"].out_dim
        in_num_features = self.layers_dict[f"convBlock{layer}"].out_features

    self.layers_dict = nn.ModuleDict(self.layers_dict)

    self.relu = nn.ReLU()

    self.ffn = nn.Sequential(
                nn.Linear(input_dim*input_dim*in_num_features, 4096), 
                nn.ReLU(),
                nn.Linear(4096, 4096), 
                nn.ReLU(),
                nn.Linear(4096, 1000),          
                nn.ReLU(),
                nn.Linear(1000, 1)
              )
    self.batch_size = batch_size

  def forward(self, imgs):

    for layer_name in self.layers_dict.keys():
      imgs = self.layers_dict[layer_name](imgs)
    imgs = imgs.view(self.batch_size, -1)
    preds = self.ffn(imgs)
    return preds.squeeze()

classification_data_iterator = DataLoader(training_classification_data, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
classification_model = DetectionModel()
classification_model.to(device)
classification_optimizer = optim.Adam(classification_model.parameters())
classification_loss = nn.BCEWithLogitsLoss()
classification_loss.to(device)


#### TRAINING MODEL STUFF #####

def train_classification_model(model, data_iterator, optimizer, loss, img_dim=IMG_DIM, batch_size=BATCH_SIZE, device=device):
  tbar = tqdm(data_iterator)
  avg_loss = []
  avg_preds = []
  bad_acc_imgs = []
  good_acc_imgs = []
  sigmoid = nn.Sigmoid()

  for batch in tbar:
    
    imgs, labels, b_img_idx = batch
    imgs = imgs.to(device).float()
    labels = labels.to(device).float()
    
    preds = model(imgs)

    b_loss = loss(preds, labels)
    b_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        b_preds = sigmoid(preds)
    
    b_preds = (b_preds > 0.5).float()
    num_correct = (b_preds == labels).sum() / len(b_preds) * 100

    if num_correct <= 65:
      bad_acc_imgs.append(b_img_idx)

    if num_correct >= 75:
      good_acc_imgs.append(b_img_idx)

    avg_loss.append(b_loss.item())
    avg_preds.append(num_correct.item())

    optimizer.zero_grad()

    tbar.set_description("b_loss - {:.4f}, avg_loss - {:.4f}, b_correct_preds - {:.2f}%, avg_correct_preds - {:.2f}%".format(b_loss, np.mean(avg_loss), num_correct, np.mean(avg_preds)))

  return avg_loss, avg_preds, bad_acc_imgs, good_acc_imgs

def test_classification_model(model, data_iterator,  loss, img_dim=IMG_DIM, batch_size=BATCH_SIZE, device=device):
  tbar = tqdm(data_iterator)
  avg_loss = []
  avg_preds = []
  sigmoid = nn.Sigmoid()

  for batch in tbar:
    
    imgs, labels, b_img_idx = batch
    imgs = imgs.to(device).float()
    labels = labels.to(device).float()
    
    with torch.no_grad():
      preds = model(imgs)

    b_loss = loss(preds, labels)

    with torch.no_grad():
        b_preds = sigmoid(preds)
    
    b_preds = (b_preds > 0.5).float()
    num_correct = (b_preds == labels).sum() / len(b_preds) * 100

    if num_correct <= 65:
      bad_acc_imgs.append(b_img_idx)

    if num_correct >= 75:
      good_acc_imgs.append(b_img_idx)

    avg_loss.append(b_loss.item())
    avg_preds.append(num_correct.item())

    tbar.set_description("b_loss - {:.4f}, avg_loss - {:.4f}, b_correct_preds - {:.2f}%, avg_correct_preds - {:.2f}%".format(b_loss, np.mean(avg_loss), num_correct, np.mean(avg_preds)))

  return avg_loss, avg_preds, bad_acc_imgs, good_acc_imgs

bad_preds = []

for batch in range(len(bad_acc_imgs)):
  imgs, labels, b_img_idx = batch
  imgs = imgs.to(device).float()
  labels = labels.to(device).float()

  with torch.no_grad():
    preds = model(imgs)

  b_loss = loss(preds, labels)

  with torch.no_grad():
    b_preds = sigmoid(preds)

  b_preds = (b_preds > 0.5).float()
  num_correct = (b_preds == labels).sum() / len(b_preds) * 100
  print(num_correct)

  not_corr_pred = (b_preds != labels)
  not_corr_img_idx = torch.masked_select(b_img_idx, not_corr_pred)
  bad_preds.append(not_corr_img_idx)


