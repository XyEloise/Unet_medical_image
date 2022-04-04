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
    #----------------------------------------------------------------------------------------------------------#
    #   video_path set path of video,video_path=0 represents camera
    #   video_save_path set path of saved video,video_save_path="" represents video won't be saved
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    
    test_interval = 100 # to test FPS

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == 'predict':
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open error')
                continue # keep  predicting
            else:
                img = unet.detect_image(img)
                img.show() # for PIL image

    elif mode == 'video':
        capture = cv2.VideoCapture(video_path)
        if video_save_path != '':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref,frame = capture.read()
        if not ref:
            raise ValueError('cannot load video')

        fps = 0.0
        while True:
            t1 = time.time()
            ref,frame = capture.read()
            if not ref:break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # image read by cv2 is BGR format
            frame = Image.fromarray(np.uint8(frame)) # unet.detect_image only support Image format
            frame = np.array(unet.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) # convert back to BGR format

            fps = (fps+(1/(time.time()-t1)))/2
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video",frame)
            if video_save_path!="":
                out.write(frame)
            if cv2.waitKey(1) & 0xff == ord('q'):break
            print("Video Detection Done!")

        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == 'fps':
        img = Image.open('img/cell.png')
        tact_time = unet.get_FPS(img,test_interval)
        print(str(tact_time)+'seconds,'+str(1/tact_time)+'FPS,@batch_size 1')

    elif mode == 'dir_predict':
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endwiths(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                img_path = os.path.join(dir_origin_path)
                img = Image.open(img_path)
                img = unet.detect_image(img)
                if not os.path.exist(dir_save_path):
                    os.makedir(dir_save_path)
                img.save(os.path.join(dir_save_path,img_name))
            
    else:
        raise AssertionError('Please specify the correct mode')



