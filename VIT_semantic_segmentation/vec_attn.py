
#### Impl of Vector Attention ####

class Vec_attention(nn.Module):
  def __init__(self, seq_len, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout=DROPOUT):
    super(Attention, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.inner_dim = self.embedding_dim * self.num_heads
    self.seq_len = seq_len

    self.to_qkv = nn.Linear(dim, self.inner_dim*3, bias=False)
    self.pos_encodings = nn.Parameters(1, self.seq_len, self.embedding_dim)

    self.attn_mlp = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.embedding_dim))

    self.output_mlp = nn.Sequential(
                        nn.Linear(self.inner_dim, self.embedding_dim),
                        nn.Dropout(dropout))

  def forward(self, x):
    b, n, _, h = *x.shape, self.heads
    qkv = self.to_qkv(x).chunk(3, dim=-1)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
    #shape of q, k, v -> (bs, nh, sl, d)

    attn_ops = k.unsqueeze(2) - q.unsqueeze(3) #shape -> (bs, nh, sl, sl, d)
    attn_ops += self.pos_encodings
    attn_ops = self.attn_mlp(attn_ops)
    attn_out = attn_ops * v.unsqueeze(3)
    attn_out = attn_out.sum(3) #shape -> (bs, nh, sl, d)
    return self.output_mlp(attn_out) #shape -> (bs, sl, d)



