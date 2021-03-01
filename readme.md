
# Glomeruli Detection and Semantic Segmentation (kaggle competition)

### Info
  1. The created dataset has 132K images
  2. Time Taken to create 132K images and labels of size 224px is 9mins
  3. Total 162K images of size 224px of which 128K are glomeruli images
  4. Something weird is happening transposing img on the axis (2,0,1) rather than reshaping inc the accuracy by 15%.


### Types of models
  1. Basic Unet
  2. Classification and then Unet architecture
  3. Transformer based Semantic segmentation


#### Ideas
  1. can Convolution 3D be added to UNet
  2. IMPORTANT reduce parameters present in the model in a strategic way because model is always overfitting.
  3. Do changing the brightness of img is really need?
  4. Should we change the size of the image?


#### Image Augmentation performed in the dataset
  1. Blurring Image with prob of (0.3)
  2. Flip all glomeruli images
  3. Mirroring all glomeruli images
  4. Changes the brightness of image between 0.8 and 1.3


#### Benchmarks for Fast data preprocessing
  1. changing the brightness of image is about  (800 micro secs)


#### Checklist for improving classification model accuracy
  - [ X ] the model was overfitting because there was to many parameters to just classify wheter an image has glomeruli or not.
  - Current model accuracy is 84% with 3 epochs with Augmentation 3 images.
  - [  ] Goal accuracy is 95%.
  - Model is not able to recognize Aug 3 images i.e. images that has different brightness.


#### VIT Model
  - can we add conv Nets to VIT? yes.
  - Instead of Dot product between Keys and queries there should be matrix multiplication i.e. use vector attention.
  - Added Conv to VIT for upsampling and got acc of 88.37% and loss of 0.27 with 2 epochs without Aug 3 Images.
  - Which loss function will be the best ?
    1. BCELossWithLogits
    2. Focal Loss
    3. Dice Loss
    4. Weighted BCELossWithLogits
 - Ideas to improve the current VIT-Conv model? -> Nothing work.
  1. Reduce Number of Filters in Conv2d.
  2. Add batchNorm2d.
  3. Change the reduce embedding dim.
  4. Added More attention layers.

  ##### Issues with VIT
  - recognising few pixels in not glomeruli image.

#### Multi Guided Conv Attention
  - Make use of Anatomical Segmentation given in the dataset
  - Model overfits after 2 epochs best prediction score 79% +- 1
  
