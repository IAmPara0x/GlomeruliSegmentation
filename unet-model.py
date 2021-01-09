

class ConvBlock(nn.Module):
  def __init__(self, in_channels, kernel_size, dilation, padding, dropout):
    super(ConvBlock, self).__init__()
    if in_channels == 3:
      self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=kernel_size, padding=padding, padding_mode="reflect")
      self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding, padding_mode="reflect")
      self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, dilation=dilation, padding=padding+(dilation-padding), padding_mode="reflect")
    else:
      self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=kernel_size, padding=padding, padding_mode="reflect")
      self.conv2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=padding, padding_mode="reflect")
      self.conv3 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, dilation=dilation, padding=padding+(dilation-padding),
                              padding_mode="reflect")

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2)
    self.dropout = nn.Dropout(dropout)

  def forward(self, b_imgs):
    b_imgs = self.conv1(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.conv2(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.conv3(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.pool(b_imgs)
    b_imgs = self.dropout(b_imgs)
    return b_imgs


class ConvTBlock(nn.Module):
  def __init__(self, in_channels, rate : int, kernel_size : int, padding : list, stride : list, output_padding=0, dilation=1, output_layer=False):
    super(ConvTBlock, self).__init__()

    if not output_layer:
      self.convTBlock = nn.ConvTranspose2d(in_channels, in_channels//rate, kernel_size=kernel_size, padding=padding[0], stride=stride[0],
                                        output_padding=output_padding)
      self.convBlock1 = nn.Conv2d(in_channels//rate, in_channels//rate, kernel_size=kernel_size, padding=padding[1],
                                    padding_mode="reflect")
      self.convBlock2 = nn.Conv2d(in_channels//rate, in_channels//rate, kernel_size=kernel_size, padding=padding[2],
                                    padding_mode="reflect")
    else:
      self.convTBlock = nn.ConvTranspose2d(in_channels, in_channels//rate, kernel_size=kernel_size, padding=padding[0], stride=stride[0],
                                        output_padding=output_padding)
      self.convBlock1 = nn.Conv2d(in_channels//rate, in_channels//rate, kernel_size=kernel_size, padding=padding[1],
                                    padding_mode="reflect")
      self.convBlock2 = nn.Conv2d(in_channels//rate, 1, kernel_size=1)
    self.relu = nn.ReLU()
    self.output_layer = output_layer
    self.sigmoid = nn.Sigmoid()

  def forward(self, b_imgs):
    b_imgs = self.convTBlock(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.convBlock1(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.convBlock2(b_imgs)
    if self.output_layer:
      return b_imgs.squeeze()
    b_imgs = self.relu(b_imgs)
    return b_imgs


class Model(nn.Module):
  def __init__(self, img_height=224, img_width=224, img_channels=3):
    super(Model, self).__init__()
    self.img_height = img_height
    self.img_width = img_width
    self.img_channels = img_channels

    self.convBlock1 = ConvBlock(img_channels, kernel_size=3, dilation=3, padding=1, dropout=0.2) #output img shape (bs, 32, 112, 112)
    self.convBlock2 = ConvBlock(32, kernel_size=3, dilation=3, padding=1, dropout=0.2)#output img shape (bs, 64, 56, 56)
    self.convBlock3 = ConvBlock(64, kernel_size=3, dilation=2, padding=1, dropout=0.1)#output img shape (bs, 128, 28, 28)
    self.convBlock4 = ConvBlock(128, kernel_size=3, dilation=2, padding=1, dropout=0.1)#output img shape (bs, 256, 14, 14)

    self.bottleneck = nn.Conv2d(256, 512, kernel_size=2, dilation=2) #shape (bs, 512, 12, 12)
    self.relu = nn.ReLU()

    self.convTBlock1 = ConvTBlock(512, rate=2, kernel_size=3, padding=[0,1,1], stride=[1, 1, 1], output_padding=0)
    self.convTBlock2 = ConvTBlock(512, rate=4, kernel_size=3, padding=[1,1,1], stride=[2, 1, 1], output_padding=1)
    self.convTBlock3 = ConvTBlock(256, rate=4, kernel_size=3, padding=[1,1,1], stride=[2, 1, 1], output_padding=1)
    self.convTBlock4 = ConvTBlock(128, rate=4, kernel_size=3, padding=[1,1,1], stride=[2, 1, 1], output_padding=1)
    self.convTBlock5 = ConvTBlock(64, rate=2, kernel_size=3, padding=[1,1,1], stride=[2, 1, 1], output_padding=1, output_layer=True)


  def forward(self, b_imgs):
    c1_imgs = self.convBlock1(b_imgs)
    c2_imgs = self.convBlock2(c1_imgs)
    c3_imgs = self.convBlock3(c2_imgs)
    c4_imgs = self.convBlock4(c3_imgs)
    b_imgs = self.bottleneck(c4_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.convTBlock1(b_imgs)
    b_imgs = torch.cat((b_imgs, c4_imgs), 1)
    b_imgs = self.convTBlock2(b_imgs)
    b_imgs = torch.cat((b_imgs, c3_imgs), 1)
    b_imgs = self.convTBlock3(b_imgs)
    b_imgs = torch.cat((b_imgs, c2_imgs), 1)
    b_imgs = self.convTBlock4(b_imgs)
    b_imgs = torch.cat((b_imgs, c1_imgs), 1)
    b_imgs = self.convTBlock5(b_imgs)
    return b_imgs


BATCH_SIZE = 64
IMG_DIM = 224
train_data_iterator = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model()
model.to(device)

optimizer = optim.Adam(model.parameters())
loss = nn.BCEWithLogitsLoss()
loss.to(device)

def train(model, train_data_iterator, optimizer, loss, img_dim=IMG_DIM, batch_size=BATCH_SIZE, device=device):
  tbar = tqdm(train_data_iterator)
  avg_loss = []
  avg_preds = []
  avg_glomeruli_correct_preds = []

  for batch in tbar:
    optimizer.zero_grad()
    imgs, labels = batch
    imgs = imgs.to(device).float()
    labels = labels.to(device).float()

    preds = model(imgs)
    labels = labels.view(batch_size, -1)
    preds = preds.view(batch_size, -1)

    b_loss = loss(preds, labels)
    b_loss.backward()
    optimizer.step()
    avg_loss.append(b_loss.cpu().item())

    b_preds = (preds > 0.5).float()
    num_correct_preds = ((b_preds == labels).sum() / (img_dim*img_dim*batch_size) ) * 100
    avg_preds.append(num_correct_preds.item())

    #### glomeruli correct preds ####
    glomeruli_idx = torch.where(labels == 1)
    glomeruli_correct_preds = ((b_preds[glomeruli_idx[0], glomeruli_idx[1]] == 1).sum() / len(glomeruli_idx[0])) * 100
    avg_glomeruli_correct_preds.append(glomeruli_correct_preds.item())


    tbar.set_description("b_loss - {:.4f}, avg_loss - {:.4f}, b_correct_preds - {:.2f}, avg_correct_preds - {:.2f}, glomeruli preds - {:.2f}, avg glom preds - {:.2f}".format(
                          b_loss, np.average(avg_loss), num_correct_preds, np.average(avg_preds), glomeruli_correct_preds, np.average(avg_glomeruli_correct_preds)))


