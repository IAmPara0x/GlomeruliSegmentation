BATCH_SIZE = 32
IMG_DIM = 224
IMG_FEATURES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 128
ATTN_HEADS = 8
DROPOUT = 0.25
HIDDEN_DIM = 256
PATCH_SIZE = 16
LAYERS = 2
CUDA_LAUNCH_BLOCKING=1

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
    self.query_layer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
    self.key_layer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
    self.value_layer = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    self.multihead_attn = nn.MultiheadAttention(self.embedding_dim, attn_heads, dropout=dropout)

  def forward(self, x):
    q = self.query_layer(x)
    k = self.key_layer(x)
    v = self.value_layer(x)

    q = rearrange(q, 'b n d -> n b d')
    k = rearrange(k, 'b n d -> n b d')
    v = rearrange(v, 'b n d -> n b d')

    x, _ = self.multihead_attn(q, k, v, attn_mask=None)
    x = rearrange(x, 'n b d -> b n d')
    return x


class FeedForward(nn.Module):
  def __init__(self, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, dim=EMBEDDING_DIM):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
              )

  def forward(self, x):
    return self.net(x)


class Model(nn.Module):
  def __init__(self, patch_size=PATCH_SIZE, img_dim=IMG_DIM, img_features=IMG_FEATURES, layers=LAYERS,
               embedding_dim=EMBEDDING_DIM, batch_size=BATCH_SIZE, dropout=DROPOUT, device=DEVICE):
    super(Model, self).__init__()
    self.patch_size = patch_size
    self.img_features = img_features
    self.seq_len = (img_features // patch_size) ** 2
    self.img_dim = img_dim
    self.embedding_dim = embedding_dim
    self.input_dim = patch_size**2*img_features
    self.output_dim = self.patch_size**2
    self.batch_size = batch_size
    self.device = device
    self.img_patch_embedding = nn.Linear(self.input_dim, self.embedding_dim)
    self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + 1, self.embedding_dim))
    self.dropout = nn.Dropout(dropout)
    # attention layers
    self.layers = nn.ModuleList([])
    for _ in range(layers):
      self.layers.append(nn.ModuleList([
        Residual(PreNorm(Attention())),
        Residual(PreNorm(FeedForward()))
        ]))
    # output layer
    self.mlp_head = nn.Sequential(
                    nn.LayerNorm(self.embedding_dim),
                    nn.Linear(self.embedding_dim, self.output_dim)
                    )
  def forward(self, x):
    #input shape of x  -> (batch_size, all_patch_len, patch_height, patch_width, 3)
    x = rearrange(x, 'b n ph pw c -> b n (ph pw c)')
    b,n,_ = x.shape
    x = self.img_patch_embedding(x) #shape (bs, pl, d)
    pos_id = torch.arange(x.size(1)).unsqueeze(0).to(self.device) #shape (pl, input_dim)
    x += self.pos_embedding[:, :(n + 1)] #shape (pl, d)
    x = self.dropout(x)
    for attn, ffn in self.layers:
      x = attn(x)
      x = ffn(x)
    x = self.mlp_head(x)
    # x output shape (bs, pl, output_dim)
    return x

#### training model ####
def train(model, train_data_iterator, optimizer, loss, img_dim=IMG_DIM, batch_size=BATCH_SIZE, device=DEVICE):
  tbar = tqdm(train_data_iterator)
  avg_loss = []
  avg_preds = []
  avg_glomeruli_correct_preds = []
  sigmoid = nn.Sigmoid()
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

    with torch.no_grad():
        b_preds_prob = sigmoid(preds)

    b_preds = (b_preds_prob > 0.5).float()
    num_correct_preds = ((b_preds == labels).sum() / (img_dim*img_dim*batch_size) ) * 100
    avg_preds.append(num_correct_preds.item())

    tbar.set_description("b_loss - {:.4f}, avg_loss - {:.4f}, b_correct_preds - {:.2f}, avg_correct_preds - {:.2f}".format(
                          b_loss, np.average(avg_loss), num_correct_preds, np.average(avg_preds)))


def check_result(idx):
  sig = nn.Sigmoid()

  img_patches, mask_patches = training_data[idx]
  img_patches = torch.FloatTensor(img_patches).to(DEVICE).unsqueeze(0)
  mask_patches = torch.FloatTensor(mask_patches)
  with torch.no_grad():
    result_mask_patches = model(img_patches)
    result_mask_patches = sig(result_mask_patches).squeeze()
  result_mask_patches  = (result_mask_patches > 0.5).float()
  img_patches = rearrange(img_patches, "(nh nw) ph pw c -> (nh ph) (nw pw) c", nh=IMG_DIM//PATCH_SIZE)
  img_patches = img_patches.cpu().numpy()
  plt.imshow(img_patches)
  result_mask_patches = rearrenge(result_mask_patches.unsqueeze(), "(nh nw) ph pw -> (nh ph) (nw pw)", nh=IMG_DIM//PATCH_SIZE)
  result_mask_patches = result_mask_patches.detach().cpu().numpy()
  mask_patches = rearrenge(mask_patches, "(nh nw) ph pw -> (nh ph) (nw pw)", nh=IMG_DIM//PATCH_SIZE)
  mask_patches = mask_patches.numpy()

  plt.imshow(result_mask_patches)
  plt.show()
  plt.imshow(mask_patches)
  plt.show()



