parameters = {}
parameters["seg_max_threshold"] = 0.5
parameters["class_min_threshold"] = 0.45
parameters["class_max_threshold"] = 0.55
parameters["class_f_preds_seg_threshold"] = 0.65
parameters["is_my_model"] = True
parameters["is_class_models"] = True

class_eff_b1_model = EfficientNet.from_pretrained("efficientnet-b1", num_classes=1)
class_eff_b1_model.float().to(DEVICE).eval()
class_eff_b1_model.load_state_dict(torch.load(class_eff_b1_model_weights))

seg_unetplusplus_model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="advprop",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
seg_unetplusplus_model.float().to(DEVICE).eval()
seg_unetplusplus_model.load_state_dict(torch.load(seg_unetplusplus_model_weights))

seg_my_model = Model()
seg_my_model.to(DEVICE).eval()
seg_my_model.load_state_dict(torch.load(seg_my_model_weights))

ensemble_seg_models = []
ensemble_seg_models.append(seg_my_model)
ensemble_seg_models.append(seg_unetplusplus_model)

ensemble_class_models = []
ensemble_class_models.append(class_eff_b1_model)

parameters["ensemble_class_models"] = ensemble_class_models
parameters["ensemble_seg_models"] = ensemble_seg_models


class Predict():
  def __init__(self, ensemble_seg_models, is_my_model=True, is_class_models=True, *args, **kwargs):
    self.ensemble_seg_models = ensemble_seg_models
    self.seg_max_threshold = kwargs["seg_max_threshold"]
    self.is_class_models = is_class_models
    if is_class_models:
      self.ensemble_class_models = kwargs["ensemble_class_models"]
      self.class_min_threshold = kwargs["class_min_threshold"]
      self.class_max_threshold = kwargs["class_max_threshold"]
      self.class_f_preds_seg_threshold = kwargs["class_f_preds_seg_threshold "]
    self.is_my_model = is_my_model

  def __call__(self, imgs, **kwargs):
    N, C, H, W = imgs.shape
    if self.is_my_model:
      s_imgs = kwargs["s_imgs"]

    if self.is_class_models:
      all_preds = np.zeros(len(self.ensemble_class_models)*N)
      for i, model in enumerate(self.ensemble_class_models):
        with torch.no_grad():
          preds = model(imgs)
          preds = torch.sigmoid(preds).cpu().numpy().squeeze()
          all_preds[N*i:(i+1)*N] = preds

      res = np.mean(all_preds)
      if self.class_min_threshold <= res <= self.class_max_threshold:
        threshold = self.class_f_preds_seg_threshold
        return self.seg_pred(imgs, s_imgs=s_imgs, threshold=threhold)
      elif self.class_min_threshold > res:
        return np.zeros((H,W))
      elif self.class_max_threshold < res:
        return self.seg_pred(imgs, s_imgs=s_imgs)
    else:
      return self.seg_pred(imgs)


  def seg_pred(self, imgs, **kwargs):
    N, C, H, W = imgs.shape
    if self.is_my_model:
      s_imgs = kwargs["s_imgs"]

    threshold = kwargs["threshold"] if "threshold" in kwargs else self.seg_max_threshold

    all_preds = torch.zeros((N*len(self.ensemble_seg_models),H,W)).cuda()

    for i, model in enumerate(self.ensemble_seg_models):
      with torch.no_grad():
        if i == 0:
          preds = model(imgs, s_imgs)
        else:
          preds = model(imgs)

        preds = torch.sigmoid(preds)
        preds[1,:,:] = torch.rot90(preds[1].squeeze(),3)
        preds[2,:,:] = torch.rot90(preds[2].squeeze(),2)
        preds[3,:,:] = torch.rot90(preds[3].squeeze(),1)
        all_preds[N*i:(i+1)*N, :, :] = preds

    all_preds = (torch.sum(all_preds, 0) >= threshold).float().squueze().cpu().numpy()

    return all_preds
