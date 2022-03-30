import numpy as np
from PIL import Image

#------------------------------#
# convert every image to RGB 
#------------------------------#
def cvtColor(image):
    """
    if len(np.shape(image)) == 3 and np.shape(image)[-1] == 3: #image is Image format
        return image 
    else:
        image = image.convert('RGB')
        return image 
    """
    #--------------------------------------------------#
    # to check since the order can be easily changed
    #--------------------------------------------------#
    if not image.mode == 'RGB':
        image = image.convert('RGB')
        return image
    else:
        return image

#---------------------------------------------------#
#   resize image
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)  #Image.BICUBIC is one way for image interpolation
    new_image = Image.new('RGB', size, (128,128,128))  #create a new image
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))  #to paste a image to another image

    return new_image, nw, nh #return image in Image format
    
def preprocess_input(image):
    image = image / 127.5 - 1
    return image