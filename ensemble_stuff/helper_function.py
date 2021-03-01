
def get_pred_mask(model, data_iterator, img_shape, device=DEVICE):
  tbar = tqdm(data_iterator)
  mask = np.zeros(img_shape)

  for batch in tbar:
    if len(batch) == 3:
      preds = model(batch[0].to(device).float(), s_imgs=batch[1].to(device).float())
    else:
      preds = model(batch[0].to(device).float())
    x1,x2,y1,y2 = [int(coords) for coords in re.sub(".jpg", "", batch[-1][0]).split("_")]
    mask[x1:x2,y1:y2] = preds

  return mask








