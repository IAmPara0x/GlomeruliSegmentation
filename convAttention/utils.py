
def normalizeImages(imgName):
    image = Image.open(imgName)
    image = np.array(image)
    normalized = (image-np.min(image))/(np.max(image)-np.min(image))
    normalized = normalized*255
    image = normalized.astype('uint8')


#### LOSS FUNCTIONS ####

