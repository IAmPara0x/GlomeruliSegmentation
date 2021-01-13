import torch
import torch.nn as nn
import torch.optim as optim

# This model gives acc of 82% on 70% of data
class Model(nn.Module):
  def __init__(self, input_dim=INPUT_DIM, input_features=INPUT_FEATURES):
    super(Model, self).__init__()
    self.net = nn.Sequential(
                nn.Conv2d(input_features, 32, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Conv2d(64, 128, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Conv2d(128, 164, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(164, 164, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.1),
        )
    self._conv_output()
    self.ffn = nn.Sequential(
                nn.Linear(self.conv_output_shape, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
        )
    self.input_dim = input_dim

  def _conv_output(self):
    rand_img = torch.randn(1, 3,self.input_dim,self.input_dim)
    output_ = self.net(rand_img)
    print(output_.shape)
    self.conv_output_shape = output_.view(-1)

  def forward(self, imgs):
    img = self.net(img)
    pred = self.ffn(img)
    return pred
