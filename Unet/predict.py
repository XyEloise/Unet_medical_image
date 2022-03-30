import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from unet import Unet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

if __name__ == '__main__':
    unet = Unet()
    mode = 'predict'
    