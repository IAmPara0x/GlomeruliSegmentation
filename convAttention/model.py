## This model is taken from https://github.com/sinAshish/Multi-Scale-Attention

class EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, dropout=False):
    super(EncoderBlock, self).__init__()
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    self.encode = nn.Sequential(*layers)

  def forward(self, x):
    return self.encode(x)


class DecoderBlock(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(DecoderBlock, self).__init__()
    self.decode = nn.Sequential(
        nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(middle_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(middle_channels),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
    )

  def forward(self, x):
    return self.decode(x)


class semanticModule(nn.Module):
  def __init__(self, in_dim):
    super(semanticModule, self).__init__()
    self.chanel_in = in_dim

    self.enc1 = EncoderBlock(in_dim, in_dim*2)
    self.enc2 = EncoderBlock(in_dim*2, in_dim*4)
    self.dec2 = DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
    self.dec1 = DecoderBlock(in_dim * 2, in_dim, in_dim )

  def forward(self,x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)

    dec2 = self.dec2(enc2)
    dec1 = self.dec1(F.upsample(dec2, enc1.size()[2:], mode='bilinear'))

    return enc2.view(-1), dec1


class PAM_Module(nn.Module):
  def __init__(self, in_dim):
    super(PAM_Module, self).__init__()
    self.chanel_in = in_dim

    self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
    self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
    self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
    self.gamma = nn.Parameter(torch.zeros(1))
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    m_batchsize, C, height, width = x.size()
    proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
    proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

    energy = torch.bmm(proj_query, proj_key)
    attention = self.softmax(energy)
    proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, height, width)

    out = self.gamma * out + x
    return out


class CAM_Module(nn.Module):
  def __init__(self, in_dim):
    super(CAM_Module, self).__init__()
    self.chanel_in = in_dim

    self.gamma = nn.Parameter(torch.zeros(1))
    self.softmax  = nn.Softmax(dim=-1)

  def forward(self, x):
    m_batchsize, C, height, width = x.size()
    proj_query = x.view(m_batchsize, C, -1)
    proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

    energy = torch.bmm(proj_query, proj_key)
    energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
    attention = self.softmax(energy_new)
    proj_value = x.view(m_batchsize, C, -1)

    out = torch.bmm(attention, proj_value)
    out = out.view(m_batchsize, C, height, width)

    out = self.gamma * out + x
    return out


class PAM_CAM_Layer(nn.Module):
  def __init__(self, in_ch, use_pam = True):
    super(PAM_CAM_Layer, self).__init__()

    self.attn = nn.Sequential(
        nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(in_ch),
        nn.PReLU(),
        PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(in_ch),
        nn.PReLU()
    )

  def forward(self, x):
    return self.attn(x)


class MultiConv(nn.Module):
  def __init__(self, in_ch, out_ch, attn = True):
    super(MultiConv, self).__init__()

    self.fuse_attn = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.PReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.PReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=1),
        nn.BatchNorm2d(out_ch),
        nn.Softmax2d() if attn else nn.PReLU()
    )

  def forward(self, x):
      return self.fuse_attn(x)


def create_resnext101():
  model = resnext50_32x4d(pretrained=True)
  model.avgpool = nn.AvgPool2d((7, 7), (1, 1))
  return model


class ResNeXt101(nn.Module):
  def __init__(self):
    super(ResNeXt101, self).__init__()
    net = create_resnext101()

    net = list(net.children())
    self.layer0 = nn.Sequential(*net[:3])
    self.layer1 = nn.Sequential(*net[3: 5])
    self.layer2 = net[5]
    self.layer3 = net[6]
    self.layer4 = net[7]

  def forward(self, x):
    layer0 = self.layer0(x)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)
    return layer4


class DAF_stack(nn.Module):
  def __init__(self):
    super(DAF_stack, self).__init__()
    self.resnext = ResNeXt101()
    self.is_training = True

    self.down4 = nn.Sequential(
        nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
    )
    self.down3 = nn.Sequential(
        nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
    )
    self.down2 = nn.Sequential(
        nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
    )
    self.down1 = nn.Sequential(
        nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
    )

    inter_channels = 64
    out_channels=64

    self.conv8_1=nn.Conv2d(64,64,1)
    self.conv8_2=nn.Conv2d(64,64,1)
    self.conv8_3=nn.Conv2d(64,64,1)
    self.conv8_4=nn.Conv2d(64,64,1)
    self.softmax_1 = nn.Softmax(dim=-1)

    self.pam_attention_1_1= PAM_CAM_Layer(64, True)
    self.cam_attention_1_1= PAM_CAM_Layer(64, False)
    self.semanticModule_1_1 = semanticModule(128)

    self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    self.pam_attention_1_2 = PAM_CAM_Layer(64)
    self.cam_attention_1_2 = PAM_CAM_Layer(64, False)
    self.pam_attention_1_3 = PAM_CAM_Layer(64)
    self.cam_attention_1_3 = PAM_CAM_Layer(64, False)
    self.pam_attention_1_4 = PAM_CAM_Layer(64)
    self.cam_attention_1_4 = PAM_CAM_Layer(64, False)

    self.fuse1 = MultiConv(256, 64, False)

    self.attention4 = MultiConv(128, 64)
    self.attention3 = MultiConv(128, 64)
    self.attention2 = MultiConv(128, 64)
    self.attention1 = MultiConv(128, 64)

    self.predict4 = nn.Conv2d(64, 1, kernel_size=1)
    self.predict3 = nn.Conv2d(64, 1, kernel_size=1)
    self.predict2 = nn.Conv2d(64, 1, kernel_size=1)
    self.predict1 = nn.Conv2d(64, 1, kernel_size=1)

  def forward(self, x):
    layer0 = self.resnext.layer0(x)
    layer1 = self.resnext.layer1(layer0)
    layer2 = self.resnext.layer2(layer1)
    layer3 = self.resnext.layer3(layer2)
    layer4 = self.resnext.layer4(layer3)

    down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
    down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
    down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
    down1 = self.down1(layer1)

    predict4 = self.predict4(down4)
    predict3 = self.predict3(down3)
    predict2 = self.predict2(down2)
    predict1 = self.predict1(down1)

    fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

    semVector_1_1,semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1),1))


    attn_pam4 = self.pam_attention_1_4(torch.cat((down4, fuse1), 1))
    attn_cam4 = self.cam_attention_1_4(torch.cat((down4, fuse1), 1))

    attention1_4=self.conv8_1((attn_cam4+attn_pam4)*self.conv_sem_1_1(semanticModule_1_1))

    semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
    attn_pam3 = self.pam_attention_1_3(torch.cat((down3, fuse1), 1))
    attn_cam3 = self.cam_attention_1_3(torch.cat((down3, fuse1), 1))
    attention1_3=self.conv8_2((attn_cam3+attn_pam3)*self.conv_sem_1_2(semanticModule_1_2))

    semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
    attn_pam2 = self.pam_attention_1_2(torch.cat((down2, fuse1), 1))
    attn_cam2 = self.cam_attention_1_2(torch.cat((down2, fuse1), 1))
    attention1_2=self.conv8_3((attn_cam2+attn_pam2)*self.conv_sem_1_3(semanticModule_1_3))

    semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
    attn_pam1 = self.pam_attention_1_1(torch.cat((down1, fuse1), 1))
    attn_cam1 = self.cam_attention_1_1(torch.cat((down1, fuse1), 1))
    attention1_1 = self.conv8_4((attn_cam1+attn_pam1) * self.conv_sem_1_4(semanticModule_1_4))

    predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
    predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
    predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
    predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

    if self.is_training:
      return torch.cat((down1, fuse1), 1),\
             torch.cat((down2, fuse1), 1),\
             torch.cat((down3, fuse1), 1),\
             torch.cat((down4, fuse1), 1), \
             semanticModule_1_4, \
             semanticModule_1_3, \
             semanticModule_1_2, \
             semanticModule_1_1, \
             predict1, \
             predict2, \
             predict3, \
             predict4
    else:
      return ((predict1 + predict2 + predict3 + predict4) / 4)



#### Training stuff ####
EPOCHS = 1
LEARNING_RATE = 1e-3


def runTraining(data, device=DEVICE):
  batch_size = BATCH_SIZE
  lr = LEARNING_RATE
  epoch = EPOCHS


  BCE_loss = nn.BCEWithLogitsLoss()
  mseLoss = nn.MSELoss()

  data_loader = DataLoader(data, batch_size=batch_size, num_workers=8)


  if torch.cuda.is_available():
    bce_loss.to(device)

  optimizer = optim.adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=false)

  for i in range(epoch):
      net.train()
      avg_loss = []
      avg_corr_preds = []
      tbar = tqdm(data_loader)
      for batch in tbar:
          images, labels = batch
          images = images.to(device).float()
          labels = labels.to(device).float()

          if images.size(0) != batch_size:
              continue

          optimizer.zero_grad()
          net.zero_grad()

          inp_enc0, \
          inp_enc1, \
          inp_enc2, \
          inp_enc3, \
          out_enc0, \
          out_enc1, \
          out_enc2, \
          out_enc3, \
          outputs0, \
          outputs1, \
          outputs2, \
          outputs3 = net(images)

          segmentation_prediction = (outputs0 + outputs1 + outputs2 + outputs3) / 4
          segmentation_prediction = segmentation_prediction.view(batch_size, -1)
          labels = labels.view(batch_size, -1)
          segmentation_loss = BCE_loss(segmentation_prediction, labels)

          lossRec0 = mseLoss(inp_enc0, out_enc0)
          lossRec1 = mseLoss(inp_enc1, out_enc1)
          lossRec2 = mseLoss(inp_enc2, out_enc2)
          lossRec3 = mseLoss(inp_enc3, out_enc3)

          lossG = segmentation_loss + 0.1 * (lossRec0 + lossRec1 + lossRec2 + lossRec3)

          lossG.backward()
          optimizer.step()

          avg_loss.append(lossG.cpu().data.numpy())

          with torch.no_grad():
            preds = torch.sigmoid(segmentation_prediction)
            preds = (preds > 0.5).float()
            corr_preds = (preds == labels).float().sum()/(IMG_DIM*IMG_DIM*BATCH_SIZE) * 100
            avg_corr_preds.append(corr_preds.item())

          tbar.set_description("b_loss - {:.4f}, avg_loss - {:.4f}, b_correct_preds - {:.2f}, avg_correct_preds - {:.2f}".format(
                                lossG, np.average(avg_loss), corr_preds, np.average(avg_corr_preds)))


