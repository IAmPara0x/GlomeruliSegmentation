
#Glomeruli Detection and Semantic Segmentation (kaggle competition)


### Info
  1. The created dataset has 132K images
  2. Time Taken to create 132K images and labels of size 224px is 9mins


#### Ideas
  1. can Conv3D be added to UNet


#### Image Augmentation performend in the dataset
  1. Blurring Image with prob of (0.3)
  2. Flip all glomeruli images
  3. Mirroring all gloeruli images
  4. Changes the brightness of image between 0.8 and 1.3


#### Bechmarks for Fast data preprocessing
  1. changing the brightness of image is about  (800 micro secs)


