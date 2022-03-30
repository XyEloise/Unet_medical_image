I have founded basic knowledge concerning image processing based on Matlab and have invented a new improved canny algorithm based on OTSU’s algorithm. Although I’m new in this area, I’m extremely interested in medical imaging and I searched many papers about breast/brain tumor detection and noticed that lots of them are based on U-Net. So currently I’m building a U-Net and due to lack of dataset a


# vgg16
## each block contains two same covolution
1.多层堆叠可以增加感受野（在vgg中，多层小卷积核可以在保持与大卷积核相同的感受野的同时，一方面保持较小参数量，一方面相当于进行了更多的非线性映射，增强网络非线性拟合能力）
  2个3*3卷积堆叠等于一个5*5卷积（一个参数18个，一个参数25个）。
2.kernel不同（初始化和梯度下降都会导致不同）权重值不同，导致每层提取不一样特征

## tf.keras.layers.Conv2D
x = layers.Conv2D(64,kernel_size=(3,3),
                      activation='relu',
                      padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02),
                      name='block_conv1')(img_imput)
				
kernel_size can also be a single integer,which means the same kernel number in each dimensions(if input is a 4d vector,2 equals to (2,2,2,2))

### formula
calculation:(W-F+2P)/S+1 (W for input,F for filter,P for padding,S for stride)

## two ways to build model
### sequential
单线性模型，即只有一个输入和一个输出，且网络是由层线性堆叠
from keras.models import Sequential,Model
from keras import layers
from keras import Input
s_model = Sequential()
s_model.add(layers.Dense(32,activation='relu',input_shape(64,)))
s_model.add(layers.Dense(16,activation='relu'))

### 函数模型API实现
input_tensor = Input(shape=(64,))
x = layers.Dense(32,,activation='relu')(input_tensor)
x = layers.Dense(16,,activation='relu')(x)
output_tensor = x
model = Model(input_tensor,output_tensor)
model.summary()

					  
# Unet
## tf.layers.Input()
用于实例化keras张量

## tf.keras.layers.Concatenate(axis=)(x)
除了连接轴外，所有张量形状相同
eg：x.shape=(2,2,5) y.shape=(2,1,5)
after concatenate：shape=(2,3,5)

## if __name__ == '__mian__'
当.py文件被直接运行时，if __name__ == '__mian__'之下的代码块将被运行，当.py文件以模块形式被导入时，if __name__ == '__mian__'之下的代码不被运行

## self.__dict__.update(self._defaults)
__dict__用于存储类对象属性的一个字典，其键为属性名(self.classes)，值是属性的值，自动化实例_default中的实例变量(不需要再用self.classes=2一个一个赋值)。

## python单引号双引号区别
无，但混合使用可以减少转义字符的使用
1.I'm a student
可以’I\'m a student'或者"I'm a student"
2.hey "good"
可以'hey "good"'或者"hey \"good\""

# predict.py
## python input()
img = input('Input image filename:')
>>> Input image filename:(需要手动输入，同时输入内容将被以string的形式赋给img变量）

## try..except..else..finally（可以任意组合）
# ！！！注意补充python异常笔记（c.biancheng.net/view/2315.html以及www.cnblogs.com/king-lps三年一梦的笔记讲的比较详细)
当try中代码出现异常，系统自动生成一个异常对象，该对象被提交给python解释器并找到合适的except块处理异常，若无则运行环境中止。
当try正常，程序执行else
无论try是否有异常，最后都要执行finally

## np.expand_dims(array,axis = 0)
如果有一张灰度图，读取后shape是(360,480),可通过这个将形状改变为满足输入(1,360,480)
axis=-1,则(360,480,1)

## cv2
cv2.imshow('1',np.array(image)) #cv2的读取必须要有名称，即输入两个参数
cv2.waitKey(0) #需要用于保持图片窗口的显示，waitkey代表读取键盘的输入(任意键)，0代表一直等待
cv2.destroyAllWindows()

## cv.VideoCapture



## tqdm库
是python的进度条库，可在python长循环中添加一个进度提示信息(跟下载安装包的进度条一样)
from tqdm import tqdm
import time
for i in tqdm(10):
	time.sleep(0.01)

## tf.config：GPU的使用与分配
1.gpus = tf.config.experimental.list_physical_devices(device_type='GPU') #查看设备上GPU的列表，可以print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True) #设置仅在需要时申请显存空间
	
2.tf.config.set_visible_devices(devices=gpus[0:2],device_type='GPU') #设置当前程序可以使用的显卡0和1(或者使用环境变量控制os.environ['CUDA_VISIBLE_DEVICES']='0,1')

3.tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1024)]) #限制消耗固定大小的显存，即建立一个显存大小1GB的虚拟GPU

4.单GPU模拟多GPU环境：tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1024),tf.config.LogicalDeviceConfiguration(memory_limit=1024)]) #在GPU0上建立两个显存2GB的虚拟GPU

## PIL和cv2
cv2.imread()
Image.open(r'Medical_Datasets/Images/0.png') #只读数据不显示(Image.show())，注意加r
这两都是只读数据不显示，其中Image.open()默认以RGB的通道读取顺序，而cv2.imread()则是BGR，因此需要这俩配合使用。
其中，Image.open得到的数据类型是Image类型（注意读取问题），而cv2.imread()得到的是np.array类型。
Image.open()只是保持了图像被读取状态，但图像真实数据没有被读取，因此如果需要对图像的元素进行操作，需要使用load()方法读取数据。

### 相互转换
1.Image->cv2
img = np.array(img)或者img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
2.cv2->Image
img = Image.fromarray(np.uint8(img))或者img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

## model.predict(x,batch_size=None,verbose=0,steps=None,callbacks=None,..)
另一个是model.predict_classes预测的是类别号,predict需要后续使用argmax()得到类别号。

## Image.new和Image.paste(在这个程序好像没啥用，PIL的resize就可以满足了)
a=Image.new('RGB', size, (256,0,0)):生成size大小的红色照片(R=256,G=0,B=0)
a.paste(image,(50,50)):把image贴到a的(50,50)的位置((x1,y1,x2,y2),x1y1是从左边和上边开始的) 

## np.argmax
### 二维


### 三维！！！


# medical_annotation.py
## random.sample()用于截取列表指定长度的随机数，但是不会改变列表本身
import random
random.seed(0)
random.sample(list,2):指定list
random.sample(range(0,9),2):不需要list，可指定一定的数字范围

## range与迭代器
