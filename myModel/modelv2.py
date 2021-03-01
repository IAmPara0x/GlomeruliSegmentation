class Encoder(nn.Module):
  def __init__(self, model="b3"):
    super(Encoder, self).__init__()

    model = EfficientNet.from_pretrained(f'efficientnet-{model}')
    layers = list(model.children())
    layers = layers[:-6]

    self.convHead = nn.Sequential(*layers[:-1])
    if model == "b0":
      self.convBlock1 = nn.Sequential(*layers[-1][0])
      self.convBlock2 = nn.Sequential(*layers[-1][1:3])
      self.convBlock3 = nn.Sequential(*layers[-1][3:5])
      self.convBlock4 = nn.Sequential(*layers[-1][5:10])
      self.convBlock5 = nn.Sequential(*layers[-1][10:])
    else:
      self.convBlock1 = nn.Sequential(*layers[-1][:2])
      self.convBlock2 = nn.Sequential(*layers[-1][2:5])
      self.convBlock3 = nn.Sequential(*layers[-1][5:8])
      if model=="b1":
        self.convBlock4 = nn.Sequential(*layers[-1][8:16])
        self.convBlock5 = nn.Sequential(*layers[-1][16:])
      else:
        self.convBlock4 = nn.Sequential(*layers[-1][8:18])
        self.convBlock5 = nn.Sequential(*layers[-1][16:])

  def forward(self, x):
    x = self.convHead(x)
    x1 = self.convBlock1(x)
    x2 = self.convBlock2(x1)
    x3 = self.convBlock3(x2)
    x4 = self.convBlock4(x3)
    x5 = self.convBlock5(x4)
    return (x1,x2,x3,x4,x5)


class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc1, enc2):
        m_batchsize, C, height, width = enc1.size()
        proj_query = self.query_conv(enc1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(enc2).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(enc2).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + enc1
        return out


class DecoderBlock(nn.Module):
  def __init__(self, f_in, f_mid, f_out, mid_layers):
    super(DecoderBlock, self).__init__()
    layers = [
        nn.Conv2d(f_in, f_mid, kernel_size=3, padding=1),
        nn.BatchNorm2d(f_mid),
        nn.ReLU(inplace=True)
        ]
    for _ in range(mid_layers):
        layers.extend([
            nn.Conv2d(f_mid, f_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(f_mid),
            nn.ReLU(inplace=True)])
    layers.append(nn.ConvTranspose2d(f_mid, f_out, kernel_size=2, stride=2))

    self.decode = nn.Sequential(*layers)

  def forward(self, x):
    return self.decode(x)


class Model(nn.Module):
  def __init__(self, img_dim=IMG_DIM, s_img_dim=S_IMG_DIM, batch_size=BATCH_SIZE):
    super(Model, self).__init__()

    self.batch_size = batch_size

    self.enc1 = Encoder()
    self.enc2 = Encoder()

    self.linear_up1 = nn.Linear(81, 64)
    self.pam_1 = PAM_Module(384)
    self.dconv1 = DecoderBlock(384,232,136,4)
    self.linear_up2 = nn.Linear(324, 256)
    self.pam_2 = PAM_Module(136)
    self.dconv2 = DecoderBlock(136*2,96,48,3)

    self.linear_up3 = nn.Linear(36**2, 32**2)
    self.pam_3 = PAM_Module(48)
    self.dconv3 = DecoderBlock(48*2,48,32,3)

    self.dconv4 = DecoderBlock(32*2,32,24,2)

    self.dconv5 = DecoderBlock(24*2, 24, 1,1)

  def forward(self, x1, x2):
    e1_x1, e1_x2, e1_x3, e1_x4, e1_x5 = self.enc1(x1)

    e2_x1, e2_x2, e2_x3, e2_x4, e2_x5 = self.enc2(x2)

    e2_x5 = self.linear_up1(e2_x5.reshape(e2_x5.shape[0], e2_x5.shape[1], -1))
    e2_x5 = e2_x5.reshape(e2_x5.shape[0], e2_x5.shape[1], 8, 8)
    x = self.pam_1(e2_x5, e1_x5)
    x = self.dconv1(x)

    e2_x4 = self.linear_up2(e2_x4.reshape(e2_x4.shape[0], e2_x4.shape[1], -1))
    e2_x4 = e2_x4.reshape(e2_x4.shape[0], e2_x4.shape[1], 16, 16)
    e1_x4_ = self.pam_2(e2_x4, e1_x4)
    x = self.dconv2(torch.cat((x, e1_x4_), 1))

    e2_x3 = self.linear_up3(e2_x3.reshape(e2_x3.shape[0], e2_x3.shape[1], -1))
    e2_x3 = e2_x3.reshape(e2_x3.shape[0], e2_x3.shape[1], 32, 32)
    e1_x3_ = self.pam_3(e2_x3, e1_x3)
    x = self.dconv3(torch.cat((x, e1_x3_), 1))

    x = self.dconv4(torch.cat((x, e1_x2), 1))
    x = self.dconv5(torch.cat((x, e1_x1), 1))

    return x, ((e1_x4, e1_x5),(e2_x4, e2_x5))

#### Segmetation Model ####
class SegModel(nn.Module):
  def __init__(self):
    super(SegModel).__init__()
    self.batch_size = batch_size

    self.enc1 = Encoder()
    self.enc2 = Encoder()

    self.pam_1 = PAM_Module(320)
    self.dconv1 = DecoderBlock(320,190,112, 2)

    self.pam_2 = PAM_Module(112)
    self.dconv2 = DecoderBlock(112*2,131,78,2)

    self.pam_3 = PAM_Module(40)
    self.dconv3 = DecoderBlock(118,70,40,1)

    self.dconv4 = DecoderBlock(64,38,22,1)

    self.dconv5 = DecoderBlock(38,16,1,1)

  def forward(self):
    e1_x1, e1_x2, e1_x3, e1_x4, e1_x5 = self.enc1(x1)

    e2_x1, e2_x2, e2_x3, e2_x4, e2_x5 = self.enc2(x2)

    x = self.pam_1(e2_x5, e1_x5)
    x = self.dconv1(x)

    e1_x4_ = self.pam_2(e2_x4, e1_x4)
    x = self.dconv2(torch.cat((x, e1_x4_), 1))

    e1_x3_ = self.pam_3(e2_x3, e1_x3)
    x = self.dconv3(torch.cat((x, e1_x3_), 1))

    x = self.dconv4(torch.cat((x, e1_x2), 1))
    x = self.dconv5(torch.cat((x, e1_x1), 1))

    return x, ((e1_x4, e1_x5),(e2_x4, e2_x5))


#### Testing stuff ####

def get_smaller_test_imgs(img_id, img, ana_mask, img_dim, s_img_dim, stride=200):
  imgheight, imgwidth, imgchannels = img.shape

  curr_h = 0
  small_img_id = 0

  diff_dim = (s_img_dim - img_dim) // 2
  count = 0

  while curr_h + img_dim <= imgheight:
    curr_w = 0
    while curr_w + img_dim <= imgwidth:
      ana_mask_arr = ana_mask[curr_h:curr_h+img_dim, curr_w:curr_w+img_dim, :]
      if np.mean(ana_mask_arr) != 0:
          smallimg_arr = img[curr_h:curr_h+img_dim, curr_w:curr_w+img_dim, :]

          #### Coords for s_img
          h1 = curr_h - diff_dim if curr_h - diff_dim > 0 else 0
          h2 = curr_h + img_dim + diff_dim

          w1 = curr_w - diff_dim if curr_w - diff_dim > 0 else 0
          w2 = curr_w + img_dim + diff_dim

          s_img = img[h1:h2, w1:w2, :]

          if s_img.shape != (s_img_dim, s_img_dim, 3):
            if s_img.shape[0] % 2 != 0:
              top = (s_img_dim - s_img.shape[0]) // 2 + 1
              bot = (s_img_dim - s_img.shape[0]) // 2
            else: top = (s_img_dim - s_img.shape[0]) // 2 bot = (s_img_dim - s_img.shape[0]) // 2

            if s_img.shape[1] % 2 != 0:
              left = (s_img_dim - s_img.shape[1]) // 2 + 1
              right = (s_img_dim - s_img.shape[1]) // 2
            else:
              left = (s_img_dim - s_img.shape[1]) // 2
              right = (s_img_dim - s_img.shape[1]) // 2
            s_img = cv2.copyMakeBorder(s_img, top,bot,left,right,cv2.BORDER_CONSTANT, value=[255,255,255])
          ####
          smallimg_arr = cv2.resize(smallimg_arr, (smallimg_arr.shape[0]//2, smallimg_arr.shape[1]//2), interpolation=cv2.INTER_AREA)
          s_img = cv2.resize(s_img, (s_img.shape[0]//2, s_img.shape[1]//2), interpolation=cv2.INTER_AREA)

          cv2.imwrite("/kaggle/temp/test_data/images/{}_{}_{}_{}.jpg".format(
                                                                        curr_h, curr_h+img_dim, curr_w, curr_w+img_dim), smallimg_arr)
          cv2.imwrite("/kaggle/temp/test_data/s_images/{}_{}_{}_{}.jpg".format(
                                                                        curr_h, curr_h+img_dim, curr_w, curr_w+img_dim), s_img)
      curr_w += stride
    curr_h += stride
  print("completed")


