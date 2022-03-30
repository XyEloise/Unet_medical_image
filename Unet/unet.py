import colorsys
import copy
import time

import cv2
import numpy as np
from PIL import Image

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image

class Unet:
    _defaults = {
        "model_path"        : 'model_data/unet_vgg_voc.h5',
        "num_classes"       : 21,
        "backbone"          : "vgg",
        "input_shape"       : [512, 512],
        #----------------------------------------------------------------#
        #   blend is for whether choose to mix edges and original image
        #----------------------------------------------------------------#
        "blend"             : True,
    }

    #---------------------------------------------------#
    #   initialize UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value) # create variables

        #---------------------------------------------------#
        #   set different colors(cause pixel-wise network)
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        #---------------------------------------------------#
        #  get model
        #---------------------------------------------------#
        self.generate()

    #---------------------------------------------------#
    #   form model
    #---------------------------------------------------#
    def generate(self):
        self.model = unet(self.input_shape[0],self.input_shape[1],self.num_classes,self.backbone)
        self.model.load_weights(self.model_path)
        print('{} model loaded'.format(self.model_path))

    #---------------------------------------------------#
    #   image detection
    #---------------------------------------------------#
    def detect_image(self,image):
        #---------------------------------------------------------#
        #   convert any image to RGB since it can only process RGB
        #---------------------------------------------------------#
        image = cvtColor(image) # use PIL cuz opencv causes inaccuracy when reading png image
        #---------------------------------------------------------#
        #   get a copy for later plotting
        #---------------------------------------------------------#
        old_image = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shapr[1]

        #---------------------------------------------------------#
        #   add gray part in case image is distorted
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[0], self.input_shape[1]))
        #---------------------------------------------------------#
        #   Normalization + add batch ndim
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image,np.float32)),0) #image_data.shape (1, 512, 512, 3)
        #---------------------------------------------------------#
        #   load image to net for prediction
        #---------------------------------------------------------------------------#
        pr = self.model.predict(image_data)[0] # use [0] is because batch dimension
        #---------------------------------------------------------------------------#
        #   get rid of gary part
        #---------------------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   resize image to original size
        #---------------------------------------------------#
        pr = cv2.resize(pr, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   retrieve classes of pixels
        #---------------------------------------------------#
            #before_np.array(pr).shape (256, 256, 2)
        pr = pr.argmax(axis=-1)
            #after_np.array(pr).shape (256, 256)
        #---------------------------------------------------------------#
        #   create a new image with colors presenting relevant classes
        '''
            pr[:,:]:
            [[1 0 0 ... 0 0 0]
            [0 0 0 ... 0 0 0]
            [0 0 0 ... 0 0 0]
            ...
            [0 0 1 ... 0 0 0]
            [1 1 1 ... 0 0 0]
            [1 1 1 ... 0 0 0]]
        '''
        #---------------------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8') #pr[:,: ] == c verifies through every number of pr
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')
        
        #------------------------------------------------#
        #   transfer to Image format
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))

        #------------------------------------------------#
        #   mix edges and original image
        #------------------------------------------------#
        if self.blend:
            image = Image.blend(old_image,image,0.7)

        return image

    def get_FPS(self,image,test_interval):
        #------------------------------------------------#
        #   pre-process image
        #------------------------------------------------#
        image       = cvtColor(image)
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
        #------------------------------------------------#
        #   test
        #------------------------------------------------#
        t1 = time.time()
        for _ in range(test_interval):
            pr = self.model.predict(image_data)[0]
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = pr.argmax(axis=-1)
        t2 = time.time()
        tact_time = (t2-t1)/test_interval # 1/tact_time is FPS
        return tact_time
