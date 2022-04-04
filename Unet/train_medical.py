import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.unet import Unet
from nets.unet_training import (CE, Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss)
from utils.callbacks import (ExponentDecayScheduler, LossHistory,
                             ModelCheckpoint)
from utils.dataloader_medical import UnetDataset
from utils.utils_fit import fit_one_epoch_no_val
from utils.utils_metrics import Iou_score, f_score

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    eager = True  # eager mode
    num_classes = 2
    backbone = 'vgg'
    model_path      = 'model_data/unet_vgg_voc.h5'
    input_shape     = [512,512]
    #----------------------------------------------------#
    #  freeze mode
    #  backbone is freezed
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #  unfreeze mode
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr         = 1e-5

    VOCdevkit_path  = 'Medical_Datasets'
    #--------------------------------------------------------------------#
    #   dice_loss：
    #   for number of classes under 10,set to True
    #   for number of classes above 10 and batch_size bigger than 10,set to True
    #   for number of classes above 10 and batch_size smaller than 10,set to True
    #---------------------------------------------------------------------# 
    dice_loss       = False
    #---------------------------------------------------------------------# 
    #   use focal loss to prevent unbalanced positive and negative samples
    #---------------------------------------------------------------------# 
    focal_loss      = False
    #---------------------------------------------------------------------# 
    #   assgin different weight to different loss
    #---------------------------------------------------------------------# 
    cls_weights     = np.ones([num_classes], np.float32)
    #---------------------------------------------------------------------# 
    #   default setting:first train without backbone then train all
    #---------------------------------------------------------------------# 
    Freeze_Train    = True
    #---------------------------------------------------------------------# 
    #   whether use multithreading to load data,1 is no
    #---------------------------------------------------------------------# 
    num_workers     = 1

    #------------------------------------------------------#
    #   get model
    #------------------------------------------------------#
    model = Unet([input_shape[0], input_shape[1], 3], num_classes, backbone)
    if model_path != '':
        model.load_weights(model_path,, by_name=True, skip_mismatch=True)

    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    #-------------------------------------------------------------------------------#
    #   set training params
    #   logging saves for tensorboard's save path
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
                        monitor = 'loss', save_weights_only = True, save_best_only = False, period = 1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.96, verbose = 1)
    early_stopping  = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    loss_history    = LossHistory('logs/', val_loss_flag = False)

    if focal_loss:
        if dice_loss:
            loss = dice_loss_with_Focal_Loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if dice_loss:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)

    freeze_layers = 17
    #-------------------------------------------------------------------#
    #   freeze backbone from training
    #   since bakbone's pre-weights are loaded
    #   freeze to avoid pre-weights being destroyed
    #   no loading and use random pre-weights result in bad performance
    #   freeze also can fasten the training
    #-------------------------------------------------------------------#
    if Freeze_Train:
        for i in range(freeze_layers):
            model.layers[i].trainable = False
        print('Freeze the first {} layers of total layers.'.format(freeze_layers,len(model.layers)))
    
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step  = len(train_lines) // batch_size
        if epoch_step == 0:
            raise ValueError("datasets too small to train")

        train_dataloader    = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)
        print('Train on {} samples, with batch size {}.'.format(len(train_lines), batch_size))

        if not eager:
            gen = tf.data.Dataset.from_gengerator(partial(train_dataloader),(tf.float32,tf.float32))
            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size) 

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch_no_val(model, loss, loss_history, optimizer, epoch, epoch_step, gen, 
                            end_epoch, f_score())

        else:
            model.compile(loss = loss,
                    optimizer = Adam(lr=lr),
                    metrics = [f_score()])

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )

    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step          = len(train_lines) // batch_size
        if epoch_step == 0:
            raise ValueError("datasets too small to train")

        train_dataloader    = UnetDataset(train_lines, input_shape, batch_size, num_classes, True, VOCdevkit_path)

        print('Train on {} samples, with batch size {}.'.format(len(train_lines), batch_size))
        if not eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32))
            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(start_epoch, end_epoch):
                fit_one_epoch_no_val(model, loss, loss_history, optimizer, epoch, epoch_step, gen, 
                            end_epoch, f_score())

        else:
            model.compile(loss = loss,
                    optimizer = Adam(lr=lr),
                    metrics = [f_score()])

            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )




        

    