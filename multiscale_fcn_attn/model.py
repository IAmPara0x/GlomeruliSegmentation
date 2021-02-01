
class Model(nn.Module): def __init__(self, unsc_img_dim=UNSC_IMG_DIM, img_features=IMG_FEATURES, sc_img_dim=SC_IMG_DIM):
    super(Model,self).__init__()

    self.unsc_img_dim = unsc_img_dim
    self.img_features = img_features
    self.sc_img_dim = sc_img_features

    self.unsc_img_net = nn.Sequential(

                          nn.Conv2d(img_features, 16, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(16, 16, 3, padding=3, dilation=3),
                          nn.ReLU(),
                          nn.BatchNorm2d(16),
                          nn.MaxPool2d(2),

                          nn.Conv2d(16, 32, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=3, dilation=3),
                          nn.ReLU(),
                          nn.BatchNorm2d(32),
                          nn.MaxPool2d(2),

                          nn.Conv2d(32, 64, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=2, dilation=2),
                          nn.ReLU(),
                          nn.BatchNorm2d(64),
                          nn.MaxPool2d(2),

                          nn.Conv2d(64, 128, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, padding=2, dilation=2),
                          nn.ReLU(),
                          nn.BatchNorm2d(128),
                          nn.MaxPool2d(2),


                          nn.Conv2d(128, 256, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(256, 256, 3, padding=2, dilation=2),
                          nn.ReLU(),
                          nn.BatchNorm2d(256),
                          nn.MaxPool2d(2))

    self.sc_img_net = nn.Sequential(
                          nn.Conv2d(img_features, 16, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(16, 16, 3, padding=3, dilation=3),
                          nn.ReLU(),
                          nn.BatchNorm2d(16),
                          nn.MaxPool2d(2),

                          nn.Conv2d(16, 32, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=3, dilation=3),
                          nn.ReLU(),
                          nn.BatchNorm2d(32),
                          nn.MaxPool2d(2),

                          nn.Conv2d(32, 64, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=3, dilation=3),
                          nn.ReLU(),
                          nn.BatchNorm2d(64),
                          nn.MaxPool2d(2),

                          nn.Conv2d(64, 128, 3, padding=1),
                          nn.ReLU(),
                          nn.Conv2d(128, 128, 3, padding=2, dilation=2),
                          nn.ReLU(),
                          nn.BatchNorm2d(128),
                          nn.MaxPool2d(2)
                      )
    self.decoder_net = nn.Sequential(
                        nn.ConvTranspose2d(384, 384, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(384, 384, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(384, 256, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, 1))

  def forward(self, unsc_img, sc_img):
    unsc_img = self.unsc_img_net(unsc_img)
    sc_img = self.sc_img_net(unsc_img)
    enc_img = torch.cat((unsc_img, sc_img), 1)
    return self.decoder_net(enc_img)


