
class Attention(nn.Module):
  def __init__(self):
    super(Attention, self).__init__()

  def forward(self):
    pass


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.encoder = EfficientNet.from_pretrained("efficientnet-b3")

  def forward(self):
    pass








