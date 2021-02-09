
#### Approx Acc of 88.5% with 2 epochs####

BATCH_SIZE = 32
IMG_DIM = 256
IMG_FEATURES = 3
EMBEDDING_DIM = 196
ATTN_HEADS = 7
DROPOUT = 0.25
HIDDEN_DIM = 128
PATCH_SIZE = 8
LAYERS = 1
CUDA_LAUNCH_BLOCKING=1
ATTN_OUTPUT_DIM = 4


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, fn, dim=EMBEDDING_DIM):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
  def __init__(self, embedding_dim=EMBEDDING_DIM, attn_heads=ATTN_HEADS, dropout=DROPOUT):
    super(Attention, self).__init__()
    self.embedding_dim = embedding_dim

    self.multihead_attn = nn.MultiheadAttention(self.embedding_dim, attn_heads, dropout=dropout)

  def forward(self, x):
    x = rearrange(x, 'b n d -> n b d')
    x, _ = self.multihead_attn(x, x, x)
    x = rearrange(x, 'n b d -> b n d')
    return x


class FeedForward(nn.Module):
  def __init__(self, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, dim=EMBEDDING_DIM):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
              )

  def forward(self, x):
    return self.net(x)

class Model(nn.Module): def __init__(self, patch_size=PATCH_SIZE, img_dim=IMG_DIM, img_features=IMG_FEATURES, layers=LAYERS,
               embedding_dim=EMBEDDING_DIM, batch_size=BATCH_SIZE, dropout=DROPOUT,
               attn_output_dim=ATTN_OUTPUT_DIM, device=DEVICE):
    super(Model, self).__init__()
    self.patch_size = patch_size self.img_features = img_features
    self.seq_len = (img_features // patch_size) ** 2
    self.img_dim = img_dim
    self.embedding_dim = embedding_dim
    self.input_dim = patch_size**2*self.img_features
    self.attn_output_dim = attn_output_dim
    self.batch_size = batch_size
    self.device = device

    self.img_patch_embedding = nn.Linear(self.input_dim, self.embedding_dim)
    self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len+1, self.embedding_dim))
    self.dropout = nn.Dropout(dropout)

    # attention layers
    self.layers = nn.ModuleList([])
    for _ in range(layers):
      self.layers.append(nn.ModuleList([
          Residual(PreNorm(Attention())),
          Residual(PreNorm(FeedForward()))
        ]))
    # output layer
    self.layernorm = nn.LayerNorm(self.embedding_dim)

    self.mlp_head = nn.Sequential(
                      nn.LayerNorm(self.embedding_dim),
                      nn.Linear(self.embedding_dim, self.attn_output_dim),
#                       nn.ReLU(),
                      )

    self.output_layer = nn.Sequential(
                        nn.ConvTranspose2d(1024, 756, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(756, 756, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(756, 512, 3, padding=1, stride=2, output_padding=1), nn.ReLU(),
                        nn.Conv2d(512, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 16, 3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 1, 1)
                      )

  def forward(self, x):
    #input shape of x -> (batch_size, all_patch_len, patch_height, patch_width, 3)
    x = rearrange(x, 'b n ph pw c -> b n (ph pw c)')

    b,n,_ = x.shape
    x = self.img_patch_embedding(x) #shape (bs, pl, d)
    x += self.pos_embedding[:, :(n+1)] #shape (pl, d)
    x = self.dropout(x)
    for attn, ffn in self.layers:
      x = attn(x)
      x = ffn(x)

    x = self.mlp_head(x)
    # x output shape (bs, seq_len, embedding_dim)
    x = rearrange(x, 'b f (h w) -> b f h w', h=int(self.attn_output_dim**(0.5))) #shape (bs, seq_len, d/2, d/2)
    x = self.output_layer(x)
    return x

#### Code for submission ####

img = tiff.imread("/kaggle/input/hubmap-kidney-segmentation/test/afa5e8098.tiff")


for i in range(0, img.shape[0] - (img.shape[0] % IMG_DIM)):
  for j in range(1, img.shape[1] - (img.shape[1] % IMG_DIM)):
    small_img = img[i:i+IMG_DIM, j:j+IMG_DIM, :]
    small_img = get_image_patches(small_img)
    small_img = torch.FloatTensor(small_img).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
      preds = model(small_img)
      preds = torch.sigmoid(preds).squeeze().cpu()
      preds = (preds > 0.5).float().numpy()

      print(i,j, end="\r")
      if np.mean(preds) > 0:
        plt.imshow(small_img)
        plt.show()
        plt.imshow(preds)
        plt.show()
        input()
        clear_output(wait=True)


def get_output(i,j):
  small_img = img[i:i+IMG_DIM, j:j+IMG_DIM, :]
  small_img = get_image_patches(small_img)
























  small_img = torch.FloatTensor(small_img).to(DEVICE).unsqueeze(0)
  with torch.no_grad():
    preds = model(small_img)
    preds = torch.sigmoid(preds).squeeze().cpu()
    preds = (preds > 0.5).float().numpy()

    if np.mean(preds) > 0:
      print(i,j)
      plt.imshow(small_img)
      plt.show()
      plt.imshow(preds)
      plt.show()

