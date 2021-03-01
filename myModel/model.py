
### Implementation of transUnet

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    model = EfficientNet.from_pretrained('efficientnet-b3')
    layers = list(model.children())
    layers = layers[:-6]
    self.convHead = nn.Sequential(*layers[:-1])
    self.convBlock1 = nn.Sequential(*layers[:-1][:2])
    self.convBlock2 = nn.Sequential(*layers[:-1][2:5])
    self.convBlock3 = nn.Sequential(*layers[:-1][5:8])
    self.convBlock4 = nn.Sequential(*layers[:-1][8:18])
    self.convBlock5 = nn.Sequential(*layers[:-1][18:])

  def forward(self, x):
    x = self.convHead(x)
    x1 = self.convBlock1(x)
    x2 = self.convBlock2(x1)
    x3 = self.convBlock3(x2)
    x4 = self.convBlock4(x3)
    x5 = self.convBlock5(x4)
    return (x1,x2,x3,x4,x5)


class DecoderBlock(nn.Module):
  def __init__(self, f_in, f_mid, f_out):
    super(DecoderBlock, self).__init__()
    self.decode = nn.Sequential(
        nn.Conv2d(f_in, f_mid, kernel_size=3, padding=1),
        nn.BatchNorm2d(f_mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(f_mid, f_mid, kernel_size=3, padding=1),
        nn.BatchNorm2d(f_mid),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(f_mid, f_out, kernel_size=2, stride=2),
    )

  def forward(self, x):
    return self.decode(x)


class Model(nn.Module):
  def __init__(self, img_dim=IMG_DIM, s_img_dim=S_IMG_DIM, batch_size=BATCH_SIZE):
    super(Model, self).__init__()

    self.batch_size = batch_size

    self.enc1 = Encoder()
    self.enc2 = Encoder()

    self.linear_up1 = nn.Linear(81, 64)
    self.bn_conv1 = nn.Conv2d(384*2, 384, 3, padding=1, bias=False)
    self.dconv1 = DecoderBlock(384,268,232)

    self.linear_up2 = nn.Linear(324, 256)
    self.bn_conv2 = nn.Conv2d(232*2, 232, 3, padding=1, bias=False)
    self.dconv2 = DecoderBlock(232*2,192,136)

    self.dconv3 = DecoderBlock(136*2,128,96)

    self.dconv4 = DecoderBlock(96*2,96,48)

    self.dconv5 = DecoderBlock(48*2, 32, 1)

  def forward(self, x1, x2):
    e1_x1, e1_x2, e1_x3, e1_x4, e1_x5 = self.enc1(x1)

    e2_x1, e2_x2, e2_x3, e2_x4, e2_x5 = self.enc2(x2)

    e2_x5 = self.linearup1(e2_x5.reshape(e2_x5.shape[0], e2_x5.shape[1], -1))
    e2_x5 = e2_x5.reshape(e2_x5.shape[0], e2_x5.shape[1], 8, 8)
    e1_x5 = self.bn_conv1(torch.cat((e2_x5, e1_x5), 1))
    e1_x5 = self.dconv2(e1_x5)

    e2_x4 = self.linearup2(e2_x4.reshape(e2_x4.shape[0], e2_x4.shape[1], -1))
    e2_x4 = e2_x4.reshape(e2_x4.shape[0], e2_x4.shape[1], 16, 16)
    e1_x4 = self.bn_conv2(torch.cat((e2_x4, e1_x4), 1))
    e1_x5 = self.dconv2(torch.cat(e1_x5, e1_x4), 1)

    e1_x5 = self.dconv3(torch.cat((e1_x5, e1_x3), 1))

    e1_x5 = self.dconv4(torch.cat((e1_x5, e1_x2), 1))

    e1_x5 = self.dconv5(torch.cat((e1_x5, e1_x1), 1))

    return e1_x5


### Training stuff

def dice_coef(output, target, is_preds=False):
    smooth = 1e-5
    if not is_preds:
      output_d = torch.sigmoid(output).view(-1).data.cpu().numpy()
      target_d = target.view(-1).data.cpu().numpy()
    else:
      output_d = output.reshape(-1)
      target_d = target.reshape(-1)
    intersection = (output_d * target_d).sum()
    return (2. * intersection + smooth) / (output_d.sum() + target_d.sum() + smooth)

def train(model, train_data_iterator, optimizer, loss, img_dim=IMG_DIM, batch_size=BATCH_SIZE, device=DEVICE):
  tbar = tqdm(train_data_iterator)
  avg_loss = []
  avg_preds = []
  avg_glomeruli_correct_preds = []
  for batch in tbar:
    optimizer.zero_grad()
    imgs, s_imgs, labels = batch
    s_imgs = s_imgs.to(device).float()
    imgs = imgs.to(device).float()
    labels = labels.to(device).float()

    preds = model(imgs, s_imgs)
    labels = labels.view(batch_size, -1)
    preds = preds.view(batch_size, -1)

    b_loss = loss(preds, labels)
    b_loss.backward()
    optimizer.step()
    avg_loss.append(b_loss.cpu().item())

    num_correct_preds = dice_coef(preds, labels)
    avg_preds.append(num_correct_preds.item())

    tbar.set_description("b_loss - {:.4f}, avg_loss - {:.4f}, b_correct_preds - {:.2f}, avg_correct_preds - {:.2f}".format(
                          b_loss, np.average(avg_loss), num_correct_preds, np.average(avg_preds)))
  return avg_loss, avg_preds


loss, preds = train(model, data_iterator, optimizer, loss)
avg_loss = [np.mean(loss[i:]) for i in range(1,len(loss)+1)]

plt.plot(avg_loss)
plt.xlabel("Batch No.")
plt.ylabel("avg loss")
plt.show()

avg_preds = [np.mean(preds[i:]) for i in range(1,len(preds)+1)]

plt.plot(avg_preds)
plt.xlabel("Batch No.")
plt.ylabel("avg preds")
plt.show()


#### Testing stuff ####

def make_grid(shape, window=IMG_DIM, min_overlap=32):
    y, x = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx*ny,4)


def save_test_imgs(model, train_df, img_idx, img_dim, s_img_dim):
  os.system("mkdir /kaggle/temp/test_data")
  os.system("mkdir /kaggle/temp/test_data/images")
  os.system("mkdir /kaggle/temp/test_data/s_images")
  img_dir = "/kaggle/temp/test_data/images"
  s_img_dir = "/kaggle/temp/test_data/s_images"

  img_id = train_df.iloc[img_idx,0]
  img = tiff.imread(f"/kaggle/input/train/{img_id}.tiff")
  slices = make_grid((img.shape[0],img.shape[1]))

  diff_dim = (s_img_dim - img_dim) // 2

  for (x1,x2,y1,y2) in slices:
    slice_img = img[x1:x2,y1:y2,:]

    #### Coords for s_img
    x1_ = x1 - diff_dim if x1 - diff_dim > 0 else 0
    x2_ = x2 + diff_dim

    y1_ = y1 - diff_dim if y1 - diff_dim > 0 else 0
    y2_ = y2 + diff_dim

    s_img = img[x1_:x2_, y1_:y2_, :]

    if s_img.shape != (s_img_dim, s_img_dim, 3):
      if s_img.shape[0] % 2 != 0:
        top = (s_img_dim - s_img.shape[0]) // 2 + 1
        bot = (s_img_dim - s_img.shape[0]) // 2
      else:
        top = (s_img_dim - s_img.shape[0]) // 2
        bot = (s_img_dim - s_img.shape[0]) // 2

      if s_img.shape[1] % 2 != 0:
        left = (s_img_dim - s_img.shape[1]) // 2 + 1
        right = (s_img_dim - s_img.shape[1]) // 2
      else:
        left = (s_img_dim - s_img.shape[1]) // 2
        right = (s_img_dim - s_img.shape[1]) // 2
      s_img = cv2.copyMakeBorder(s_img, top,bot,left,right,cv2.BORDER_CONSTANT, value=[255,255,255])
    ####
    cv2.imwrite(f"{img_dir}/{x1}_{x2}_{y1}_{y2}.jpg", slice_img)
    cv2.imwrite(f"{s_img_dir}/{x1}_{x2}_{y1}_{y2}.jpg", s_img)


def test_model(model, data_iterator, train_df, img_idx, img_shape, device=DEVICE):
  model.eval()
  tbar = tqdm(train_data_iterator)

  real_mask = rle2mask(train_df.iloc[img_idx, 1], shape=(img_shape[1], img_shape[0]))
  mask = np.zeros(img_shape)

  for batch in tbar:
    imgs, s_imgs, img_name = batch
    s_imgs = s_imgs.to(device).float()
    imgs = imgs.to(device).float()

    x1,x2,y1,y2 = [int(coords) for coords in re.sub(".jpg", "", img_name).split("_")]

    with torch.no_grad():
      preds = model(imgs, s_imgs)
      preds = torch.sigmoid(preds).cpu().squeeze()
      preds = (preds > 0.5).float().numpy()
      mask[x1:x2,y1:y2] = preds
  val = dice_coef(mask, real_mask, is_preds=True)

  print(f"dice coef : {val}")

  real_mask = cv2.resize(real_mask, (img_shape[0]//20, img_shape[1]//20))
  mask = cv2.resize(mask, (img_shape[0]//20, img_shape[1]//20))

  plt.imshow(real_mask)
  plt.show()
  plt.imshow(mask)
  plt.show()

#loss 0.975 approx


