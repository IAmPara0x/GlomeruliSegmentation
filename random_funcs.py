
def check_result(idx):
  """
  This Function Shows the output from the Conv neural net for semantic segmentation given idx.
  """
  sig = nn.Sigmoid()

  img_patches, mask_patches = training_data[idx]
  img_patches = torch.FloatTensor(img_patches).to(DEVICE).unsqueeze(0)
  mask_patches = torch.FloatTensor(mask_patches)
  with torch.no_grad():
    result_mask_patches = model(img_patches)
    result_mask_patches = sig(result_mask_patches).squeeze()
  result_mask_patches  = (result_mask_patches > 0.5).float()

  img_patches = img_patches.view(64, 64, 3).cpu().numpy()
  result_mask_patches = rearrange(result_mask_patches.squeeze(), "(nh nw) (ph pw) -> (nh ph) (nw pw)", nh=16, ph=4)
  result_mask_patches = result_mask_patches.detach().cpu().numpy()
  mask_patches = mask_patches.numpy()


  plt.imshow(img_patches/255)
  plt.show()
  plt.imshow(result_mask_patches)
  plt.show()
  plt.imshow(mask_patches)
  plt.show()


def check_result(idx):
  """
  This Function Shows the output from the VIT neural net for semantic segmentation given idx.
  """
  sig = nn.Sigmoid()

  img_patches, mask_patches = training_data[idx]
  img_patches = torch.FloatTensor(img_patches).to(DEVICE).unsqueeze(0)
  mask_patches = torch.FloatTensor(mask_patches)
  with torch.no_grad():
    result_mask_patches = model(img_patches)
    result_mask_patches = sig(result_mask_patches).squeeze()
  result_mask_patches  = (result_mask_patches > 0.5).float()


  img_patches = rearrange(img_patches.squeeze().cpu().numpy(), "(nh nw) ph pw c -> (nh ph) (nw pw) c", nh=int(img_patches.shape[1]**0.5))
  result_mask_patches = rearrange(result_mask_patches.squeeze(), "(nh nw) (ph pw) -> (nh ph) (nw pw)",
                                              nh=int(result_mask_patches.shape[0]**0.5), ph=PATCH_SIZE)

  mask_patches = rearrange(mask_patches.squeeze(), "(nh nw) ph pw -> (nh ph) (nw pw)",
                                              nh=int(mask_patches.shape[0]**0.5))
  result_mask_patches = result_mask_patches.detach().cpu().numpy()
  mask_patches = mask_patches.numpy()


  plt.imshow(img_patches/255)
  plt.show()
  plt.imshow(result_mask_patches)
  plt.show()
  plt.imshow(mask_patches)
  plt.show()

