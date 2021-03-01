BATCH_SIZE = 32
IMG_DIM = 224
IMG_FEATURES = 3
EMBEDDING_DIM = 56
ATTN_HEADS = 8
DROPOUT = 0.25
HIDDEN_DIM = 112
PATCH_SIZE = 8
LAYERS = 1
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
    self.input_dim = patch_size**2*self.img_features
    self.output_dim = 1
    self.batch_size = batch_size
    self.device = device

    self.img_patch_embedding = nn.Linear(self.input_dim, self.embedding_dim)
    self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.embedding_dim))
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
    # self.mlp_head = nn.Sequential(
    #                   nn.LayerNorm(self.embedding_dim),
    #                   nn.Linear(self.embedding_dim, self.output_dim))
    self.output_layer = nn.Sequential(
                        nn.ConvTranspose2d(self.seq_len, 256, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256, 64, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 8, 3, padding=1, stride=2, output_padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(8, 1, 2, stride=1)
                      )

  def forward(self, x):
    #input shape of x -> (batch_size, all_patch_len, patch_height, patch_width, 3)
    x = rearrange(x, 'b n ph pw c -> b n (ph pw c)')

    b,n,_ = x.shape
    x = self.img_patch_embedding(x) #shape (bs, pl, d)
    x += self.pos_embedding[:, :(n)] #shape (pl, d)
    x = self.dropout(x)
    for attn, ffn in self.layers:
      x = attn(x)
      x = ffn(x)
    x = self.layernorm(x)
    # x output shape (bs, seq_len, embedding_dim)
    x = rearrange(x, 'b f (h w) -> b f h w', h=int(self.embedding_dim/2)) #shape (bs, seq_len, d/2, d/2)
    x = self.output_layer(x)
    assert x.shape[-1] == 224
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
  return avg_loss, avg_preds


#### Evaluating model ####

def eval(model, data_iterator):
  tbar = tqdm(data_iterator)
  avg_preds = []
  sig = nn.Sigmoid()

  for batch in tbar:
    imgs, labels = batch
    imgs = imgs.to(DEVICE).float()

    with torch.no_grad():
      preds = model(imgs)
      preds = sig(preds)
      preds = preds.view(preds.shape[0], -1)
      preds = (preds > 0.5).float()

    labels = labels.view(preds.shape[0], -1)
    glom_mask = (labels == 1)
    corr_preds = (torch.masked_select(preds, glom_mask).sum() / labels.sum()) * 100
    avg_preds.append(corr_preds.item())
    tbar.set_description("avg_preds - {:.4f}, curr_preds - {:.4f}".format(np.mean(avg_preds), corr_preds))


