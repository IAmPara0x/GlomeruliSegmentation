class ConvBlock(nn.Module):
  def __init__(self, in_features, out_features, kernel_size, padding, dropout, pool_size):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
    self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=kernel_size, padding=padding)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(pool_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, b_imgs):
    b_imgs = self.conv1(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.conv2(b_imgs)
    b_imgs = self.relu(b_imgs)
    b_imgs = self.pool(b_imgs)
    b_imgs = self.dropout(b_imgs)
    return b_imgs


class ConvTBlock(nn.Module):
  def __init__(self, in_features, out_features : int, kernel_size : int, padding : int, stride : int, num_convT_layer=2, convTpadding = 1, output_padding=1):
    super(ConvTBlock, self).__init__()
    if num_convT_layer == 2:
        self.convTBlock1 = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, padding=convTpadding,
                                            output_padding=output_padding, stride=stride)
        self.convTBlock2 = nn.ConvTranspose2d(out_features, out_features, kernel_size=kernel_size, padding=convTpadding,
                                            output_padding=output_padding, stride=stride)
    else:
        self.convTBlock1 = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, padding=convTpadding,
                                            output_padding=output_padding, stride=stride)

    self.num_convT_layer = num_convT_layer
    self.convBlock1 = nn.Conv2d(out_features, out_features, kernel_size=kernel_size, padding=padding)
    self.relu = nn.ReLU()

  def forward(self, b_imgs):
    b_imgs = self.convTBlock1(b_imgs)
    b_imgs = self.relu(b_imgs)
    if self.num_convT_layer == 2: b_imgs = self.convTBlock2(b_imgs)
        b_imgs = self.relu(b_imgs)
    print(b_imgs.shape)
    b_imgs = self.convBlock1(b_imgs)
    b_imgs = self.relu(b_imgs)
    print(b_imgs.shape)
    return b_imgs


class SegmentationModel(nn.Module):
  def __init__(self, input_features=INPUT_FEATURES, input_dim=INPUT_DIM):
    super(SegmentationModel, self).__init__()
    self.convBlock1 = ConvBlock(in_features=input_features, out_features=32, kernel_size=3, padding=1, dropout=0.2, pool_size=2)
    self.convBlock2 = ConvBlock(in_features=32, out_features=64, kernel_size=3, padding=1, dropout=0.2, pool_size=4)
    self.convBlock3 = ConvBlock(in_features=64, out_features=128, kernel_size=3, padding=1, dropout=0.2, pool_size=4)
    self.bottleneck = nn.Conv2d(128, 164, 3)

    self.convTBlock1 = ConvTBlock(in_features=164, out_features=128, kernel_size=3, convTpadding=0, padding=1,
                                  num_convT_layer=1, stride=1, output_padding=0)
    self.convTBlock2 = ConvTBlock(in_features=256, out_features=64, kernel_size=3, padding=1, stride=2)
    self.convTBlock3 = ConvTBlock(in_features=128, out_features=32, kernel_size=3, padding=1, stride=2)
    self.convTBlock4 = ConvTBlock(in_features=64, out_features=32, kernel_size=3, padding=1, num_convT_layer=1, stride=2)
    self.input_features = input_features
    self.relu = nn.ReLU()
    self.outputlayer = nn.Conv2d(32, 1, 1)

  def forward(self, imgs):
    imgs1 = self.convBlock1(imgs)
    imgs2 = self.convBlock2(imgs1)
    imgs3 = self.convBlock3(imgs2)
    imgs = self.bottleneck(imgs3)
    imgs = self.relu(imgs)
    print(imgs.shape)
    imgs = self.convTBlock1(imgs)
    imgs = torch.cat((imgs3, imgs), 1)
    imgs = self.convTBlock2(imgs)
    imgs = torch.cat((imgs2, imgs), 1)
    imgs = self.convTBlock3(imgs)
    imgs = torch.cat((imgs1, imgs), 1)
    imgs = self.convTBlock4(imgs)
    imgs = self.outputlayer(imgs)
    return imgs.squeeze()
