import torch
import torch.nn as nn
import torch.optim as optim

# This model gives acc of 84% acc with 3 epochs
class DetectionModel(nn.Module):
  def __init__(self, input_dim=INPUT_DIM, input_features=INPUT_FEATURES):
    super(Model, self).__init__()
    self.net = nn.Sequential(
                nn.Conv2d(input_features, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Dropout(0.2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4)
            )

    self.input_dim = input_dim
    self._conv_output()
    self.ffn = nn.Sequential(
                nn.Linear(len(self.conv_output_shape), 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )

  def _conv_output(self):
    rand_img = torch.randn(1, 3, self.input_dim, self.input_dim)
    output_ = self.net(rand_img)
    print(output_.shape)
    self.conv_output_shape = output_.view(-1)

  def forward(self, imgs):
    imgs = self.net(imgs)
    imgs = imgs.view(imgs.shape[0], -1)
    preds = self.ffn(imgs)
    return preds.squeeze()

