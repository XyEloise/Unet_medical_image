I have founded basic knowledge concerning image processing based on Matlab and have invented a new improved canny algorithm based on OTSU’s algorithm. Although I’m new in this area, I’m extremely interested in medical imaging and I searched many papers about breast/brain tumor detection and noticed that lots of them are based on U-Net. So currently I’m building a U-Net and due to lack of dataset a

<font face="宋体" color=black size=10>Unet笔记</font>
[toc]

# vgg16
## each block contains two same covolution
1.多层堆叠可以增加感受野（在vgg中，多层小卷积核可以在保持与大卷积核相同的感受野的同时，一方面保持较小参数量，一方面相当于进行了更多的非线性映射，增强网络非线性拟合能力）
  2个3*3卷积堆叠等于一个5*5卷积（一个参数18个，一个参数25个）。
2.kernel不同（初始化和梯度下降都会导致不同）权重值不同，导致每层提取不一样特征

## tf.keras.layers.Conv2D
x = layers.Conv2D(64,kernel_size=(3,3), #64个大小为3*3的k
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
s_model = Sequential([layers.Dense(256,activation='relu')])
s_model.add(layers.Dense(32,activation='relu',input_shape(64,)))
s_model.add(layers.Dense(16,activation='relu')) #定义输出节点16
s_model.build(input_shape=(2,4)) #创建wb张量，输入两个样本，每个样本特征长度4
s_model.summary() #打印网络结构和参数量
s_model.bias
s_model.kernel

### 函数模型API实现
input_tensor = Input(shape=(64,))
x = layers.Dense(32,,activation='relu')(input_tensor)
x = layers.Dense(16,,activation='relu')(x)
output_tensor = x
model = Model(input_tensor,output_tensor)
model.summary()

					  
# Unet.py
## tf.layers.Input()
用于实例化keras张量

## tf.keras.layers.Concatenate(axis=)(x)
除了连接轴外，所有张量形状相同
eg：x.shape=(2,2,5) y.shape=(2,1,5)
after concatenate：shape=(2,3,5)

## if __name__ == '__mian__'
当.py文件被直接运行时，if __name__ == '__mian__'之下的代码块将被运行，当.py文件以模块形式被导入时，if __name__ == '__mian__'之下的代码不被运行

## self.__dict__.update(self._defaults)
__dict__用于存储类对象属性的一个字典，其键为属性名(self.classes)，值是属性的值，自动化实例_default(字典)中的实例变量(不需要再用self.classes=2一个一个赋值)。

## setattr(a,'bar',5):设置a对象的bar属性值为5
def __init__(self, **kwargs):
    for name, value in kwargs.items():
        setattr(self, name, value) 

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

## str.endswith(suffix,start,end)
判断字符串中某段字符是否以指定字符或子字符串结尾，suffix可以是单个字符、字符串或元组(由字符串组成的元组)

## try..except..else..finally（可以任意组合）
# ！！！注意补充python异常笔记（c.biancheng.net/view/2315.html以及www.cnblogs.com/king-lps三年一梦的笔记讲的比较详细)
当try中代码出现异常，系统自动生成一个异常对象，该对象被提交给python解释器并找到合适的except块处理异常，若无则运行环境中止。
当try正常，程序执行else
无论try是否有异常，最后都要执行finally

## np.expand_dims(array,axis = 0)
如果有一张灰度图，读取后shape是(360,480),可通过这个将形状改变为满足输入(1,360,480)
axis=-1,则(360,480,1)

## cv2
cv2.imshow('1',np.array(image)) #cv2的读取必须要有名称，即输入两个参数，此处image是Imgae格式的
cv2.waitKey(0) #需要用于保持图片窗口的显示，waitkey代表读取键盘的输入(任意键)，0代表一直等待
cv2.destroyAllWindows('1') #销毁所有窗口或指定窗口(cv2读取的名字)

### cv2.imread(filepath,flags)
flags:读入图片标志
1.cv2.IMREAD_COLOR:默认参数，读取彩色图片，不包括alpha通道(取值0到1，用于存储这个像素是否对图片有贡献，0代表透明，1代表不透明，不能使图片变透明，但其代表的数值可以与其他数值做运算来决定哪里透明)
2.cv2.IMREAD_GRAYSCALE:读取灰色图片
3.cv2.IMREAD_UNCHANGED:读取原图片，包括alpha通道

### cv2.imwrite(file,img,num) #保存图像
1.cv2.imwrite('0.png',img,[int(cv2.IMWRITE_JPEG_QUALITY),95]):对于img是JPEG格式，num表示的是图像的质量用0~100表示，默认95，cv2.IMWRITE_JPEG_QUALITY类型为long必须转成int
2.cv2.imwrite('0.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9]):对于img是png格式，num表示的是压缩级别，默认3，级别越高图像越小。
可以写成cv2.imwrite('F:/../test'+str(num)+'.png',img) #注意路径写法

### img.copy()(或者copy.copy(img))和copy.deepcopy(img)
#### python可变对象和不可变对象
id内存地址不变的情况下，value可以改变，称为可变类型：list、dic、set
value改变，id改变，为不可变类型：str、tuple、int、bool、float

#### 浅拷贝和深拷贝：
在python中，对象赋值并没有拷贝这个对象，而是拷贝了这个对象的引用
1.直接赋值，默认浅拷贝传递对象的引用，原始列表改变，被赋值的对象也会改变，且id始终相同
2.浅拷贝，没有拷贝子对象，共用一个对象，故原始数据改变，子对象改变，浅拷贝有两种情况：
(1)当复制的值是不可变对象时，id相同，由于是不可变类型，改变值后，只有该id和值改变，直接赋值对象和浅拷贝对象id和值不变
(2)当是可变对象时，有两种情况：
	1.复制的对象中无复杂子对象，改变值后，该值和直接赋值对象的id和值一起改变，浅拷贝对象的id和值不变
	2.若改变的对象含有复杂子对象(可变类型内嵌可变类型，直接赋值对象始终跟原对象一致，例如a=[1,2,['a','b']]中的['a','b'])，改变a[2]复杂子对象的值,浅拷贝对象值变化且id不变，改变a(a还是可变类型)，浅拷贝对象值不变且id不变。因为浅拷贝复杂子对象的保存方式是以引用方式存储的(即=)，故修改浅拷贝的值和原来的值都会改变复杂子对象的值。
3.深拷贝，拷贝对象是一个新的个体，怎么样改变id都跟原来不同，值不会被影响。

## opencv视频处理
### 调用摄像头
1.vc = cv2.VideoCapture(0):0表示打开内置摄像头，或者是视屏文件路径
2.ret,frame = vc.read():按帧读取视频,ret是布尔类型，正确读取则返回True，读取失败或读取视屏结尾会返回False，frame是每一帧的图像，为BGR格式
3.cap.isOpened()判断视频对象是否成功获取，返回bool值
4.cv2.namedWindow()：在cv2.imshow()前面加一句cv2.namedWindow('image',0)就可以自己拉伸图像框
5.if cv2.waitkey(1) & 0xFF == ord('q'):break #waitkey(1)的1代表按键输入之前的有效时间，单位毫秒，在此间按键不会被记录，并在下一次if时作用，若waitkey(0)代表程序一直处在if语句直到按键q被按下，与0xFF(11111111)相与是因为cv2.waitKey(1)的返回值不止8位，但只有后八位有效，为避免产生干扰，将其余位置0，ord()将字符转化为对应ASCII码
6.vc.release():释放摄像头资源
7.out.release()
8.cv2.destroyAllWindows()

### 录制视频并保存
1.fourcc = cv2.VideoWriter_fourcc(*'XVID') #用于设置需要保存的视频的格式
2.size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))) #获得原视频的大小
3.fps = vc.get(cv2.CAP_PROP_FPS) #获得原视频码率
4.out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size) #指定写视频的格式，video_fps为20是正常，小于则是慢镜头
5.out.write(frame) #写入图片
6.frame = cv2.putText(img,str(i),(123,456),font,2,(0,255,0),3) #依次是图片，文字，左上角坐标，字体，字体大小，颜色，字体粗细

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

## np.argmax(a,axis=None,out=None)取出最大元素对应索引值
### 二维
axis=0按列比较，axis=1按行比较，如果是None就是平铺成一维数组
np.argmax(np.array([[1,5,5,2],
                    [9,6,2,8],
					[3,7,9,1]]),axis=0)
1.当axis=0时
输出[1,2,2,1](对应第一列的9，第二列的7...)
2.当axis=1时
输出[1,0,2]

### 三维！！！(有图可看csdn作者zxy2847225301的argmax文章)
np.argmax(np.array([[[1,5,5,2],
                     [9,-6,2,8],
					 [-3,7,-9,1]
					],
					[[-1,7,-5,2],
                     [9,6,2,8],
					 [3,7,9,1]
					],
					[[21,6,-5,2],
                     [9,36,2,8],
					 [3,7,79,1]
					]
				   ]),axis=0):
1.当axis=0时
即在a[0]方向上找最大值，即三个二维矩阵作比较(即三个大元素，需要比较的是大元素)，结果也是一个二维矩阵
[[2,1,0,0]
 [0,2,0,0]
 [1,0,2,0]]
2.当axis=1时
即在a[1]方向上找最大值，即三对由三个大元素组成的一维矩阵作比较，结果也是三个一维矩阵组成的二维矩阵
[[1,2,0,1]
 [1,0,2,1]
 [0,1,2,1]]
3.当axis=2时
即在a[2]方向上找最大值，大元素是每个数字，输出结构与原矩阵相同，但是由于冗余去掉最外面中括号，还是一个二维矩阵
[[[1],[0],[1]]
 [[1],[0],[2]]
 [[0],[1],[2]]]
也即
 [[1,0,1]
  [1,0,2]
  [,1,2]]


# medical_annotation.py
## random.sample()用于截取列表指定长度的随机数，但是不会改变列表本身
import random
random.seed(0)
random.sample(list,2):指定list
random.sample(range(0,9),2):不需要list，可指定一定的数字范围

## 划分数据集例子
### 一种如medical_annotation.py所示
### 另一种
def split_train(data,test_ratio):
	np.random.seed(0)
	shuffled_indices=np.random.permutation(len(data)) #np.random.permutation打乱顺序并返回列表
	test_set_size = int(test_ratio*len(data))
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data[test_indices],data[train_indices]
	
## python读写文件
### 读文件
with open('path','r') as f:
	f.read() #read每次读取整个文件，readlines将文件内容分析成由行组成的列表，readline每次只读一行
	
### 写文件
with open('path','w') as f:
	f.write() #将字符串写入文件，writelines接收一个字符串列表并将之写入文件中(不是以列表的形式)

### os.path
1.os.path.split
2.os.path.splitext
3.os.rename('old.txt','new.py')
4.os.remove('old.txt')
5.os.path.abspath('.')查看当前目录的绝对路径
6.os.path.join
7.os.mkdir
8.os.rmdir 

## range与迭代器


# train_medical.py
## batch、step和epoch
epoch：一个epoch表示所有训练样本训练一遍
step/iteration(迭代)：每运行一次step更新一次网络参数
batch_size：一次迭代所使用样本数
step/iteration=样本数*eopch/batch_size

### mini-batch的随机梯度算法，mini-batch不需要遍历全部样本，适用于数据量非常大时
def SGD(self,training_data,epochs,mini_batch_size,eta):
	n = len(training_data)
	for i in range(epochs): #主循环走一遍所有数据
		random.shuffle(training_data)
		mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
		for mini_batch in mini_batchs: #计算随机梯度，更新权重和偏置，eta是学习率
			self.update_mini_batch(mini_batch,eta)

## 制作voc格式数据集
先使用labelme进行标注，然后转化成voc格式(大多使用的格式)
voc格式：是xml格式的

## python函数返回一个函数内部定义的函数
！！！详细的在https://zhihu.com/question/25950466/answer/31731502关于闭包装饰器的内容很详细！！！
def a():
	b = range(2)
	def c():
		return '1'
	return c
上述是一个Lazy Evaluation的例子，一般局部变量在函数返回时会被垃圾回收器回收，而使用a函数进行包装后，b变量会包含在a函数的执行环境中(因为闭包会将内层函数执行的整个环境打包)，延长了生命周期

## strip和split
str.strip(rm):删除str字符串开头和结尾处的rm字符，当rm为空，默认删除空白符('\n','\r','\t'等)
str.split('.',1)：按照'.'分割一次，一般默认n次
python中没有字符类型，只有字符串！

## 模型装配、训练与测试 p175
创建网络后，首选通过循环迭代数据集多变，每次按批产生训练数据，前向计算，然后通过损失函数计算误差值，并反向传播自动计算梯度，每iteration更新一次网络参数。

### 首先进行模型装配，通过compile指定网络优化器对象，损失函数和评价指标
from tensorflow.keras import optimizers,losses
model.compile(optimizer=optimizers.Adam(lr=0.01),loss=losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model的这个loss参数可以传入自定义loss函数，metrics也可以传入自定义，它们都必须使用tensorflow.keras.backend，因为处理和输出的都必须是张量，不需要转换Numoy类型的数据
梯度下降的反向传播使用loss指标进行(loss指标必须可导，查看反向传播原理)，但在评估模型的性能时，需要查看metrics，一方面在训练时可直观要评价的指标变化情况，一方面是可以加入EarlyStopping，当指标不再下降就停止训练

### 模型装配完后，通过fit()送入待训练数据和验证数据
(1)model.fit
history = model.fit(x_train,label_train,batch_size=32,epochs=500,callbacks,validationz_split=0.2,validation_freq=20，shuff=True) #validationz_split的划分在shuffle前，需要打乱数据集后再使用
history.history #打印损失函数等在训练中随每次epoch数值变化的记录
因为fit一次性把所有xy加载进去，当数据量很大会导致内存泄漏，这时使用fit_generator。
(2)model.fit_generator(generator,steps_per_epoch,epochs,initial_epoch,use_multiprocessing,workers,callbacks)
利用python生成器，逐个生成数据的batch并进行训练，优点是生成器与模型将并行执行以提高效率(允许在CPU上实时进行数据提升和在GPU上进行模型训练)。
steps_per_epoch:一个epoch的迭代次数，即每个epoch中生成器执行生成数据的次数，一旦达到就进入下个epoch
workers:最大进程数
use_multiprocessing：使用多线程读取数据
initial_epoch：从指定的epoch开始训练，在继续之前的训练时有用

！！！注意：keras提供了一个ImageDataGenerator类，可以实现图片生成器给fit_generator+对每个batch的图片进行数据增强

### 模型测试
(1)model.predict(x,batch_size=32,verbose=0)
按照batch获得输入数据对应的输出，只输出结果。
x,y = next((iter(test)) #加载一个batch的测试数据
out = model.predict(x)

(2)loss,accuracy = model.evaluate(x,label,batch_size=32,verbose=1)
返回一个测试误差的标量值(输出损失和精确度)，用于测试性能。
verbose:显示日志


## 梯度更新 p41 p53

## 反向传播 p167


## functools.partial偏函数
帮助我们扩展原有函数的功能，避免重复传递固定参数值，偏函数可以为之赋默认值，功能就是为某个函数绑定函数参数
functools.partial(func,*args,**kwargs):返回一个类func的函数
func:需要被扩展的参数
*args:需要被固定的未知参数，扩展的会附加进去
**kwargs:如果原来func中关键字参数不在则扩展，如果在则覆盖

## tf.data.Dataset
该数据集对象方便实现多线程、预处理、shuffle和train on batch等常用数据集功能。
train_db=tf.data.Dataset.from_tensor_slices((x,y)) #from_tensor_slices一次性加载所有数据
1.train_db.shuffle(buffer_size):防止每次训练时数据按固定顺序使得网络尝试记住顺序，可通过shuffle().step2().step3()的方式完成所有数据处理步骤
2.train_db.batch(batch_size):为利用显卡并行计算能力，网络计算过程会同时计算多个样本，为一次能够从Dataset对象中产生batchsize数量的样本，需要设置Dataset为批训练方式
3.train_db.map(preprocess):自定义preproess函数对数据进行自定义处理，使用map函数调用该函数
4.

### tf.data
tf.data的输入pipeline有三个步骤ETL：extract(从内存、磁盘、云端、远程服务器等读取数据)、transform(对数据进行处理)、load(载入GPU或CPU)
！！！根据https://blogs.csdn.net/qq_38742161/details/88399122总结

### tf.data.Dataset.from_generator(partial(train_dataloader.generate)
通过建立一个读取数据的生成器，通过tf.data.Dataset对其进行包装转换(使用partial可以建立一个新的函数从而不影响原来函数)，即可实现逐batch读入数据的目的，这时一个无尽头的生成器构建的Dataset故会一直读取数据

## 冻结特定网络层的训练
model.layers是一个列表，尽量不要使用layers.Conv2D(filters,(1,1),trainable=False)(input_tensor)，容易出错
model.layers[i].trainable = False

# callbacks.py
## tensorflow.keras.callbacks回调函数，可在训练阶段查看模型内部状态
可以传递一个列表的回调函数(作为callbacks关键字参数)到Seqential或model的fit方法，每个epoch/step/batch结束时，相应的回调函数的方法就会被在各自的阶段被调用
例如：
callbacklist = [TensorBoard(log_dir = 'logs/'),ModelCheckpoint(),EarlyStopping()]
model.fit(x,y,epochs=5,callbacks=callbacklist)

### tensorflow.keras.callbacks
是一个抽象类，只能被继承，不能实例化且子类必须要有父类所有方法且名称一样。
未绑定方法：通过使用未绑定方法(通过类名调用实例方法，实例方法是除静态和类方法之外的所有方法)可以在子类中再次调用父类中被重写的方法。
！！！super方法调用父类数据属性和函数属性：python要求如果子类重写父类的构造方法(__init__)，那么子类必须调用父类构造方法，有两种方式：1.super(ModelCheckpoint, self).__init__()2.使用未绑定方法，这种方式很容易理解。因为构造方法也是实例方法，当然可以通过这种方式来调用。

#### 创建一个回调函数
kears.callbacks.Callback用于组件新的回调函数的抽象基类，回调函数以logs字典为参数(logs数据由fit方法给出)，该字典包含一系列关于epoch和batch的信息。
包含两个属性，子类可以直接调用父类属性的值：params和model(可使用self.params和self.model进行调用，params是训练参数的字典，包含batch size等；model是keras.models.Model的实例，即正在被训练模型，该模型的属性也可被调用)
model的fit方法会在传入到回调函数的logs(传入到回调函数的logs即自定义回调函数可以使用的字典参数，用logs.get('')即可获取)里包含以下数据：on_epoch_end(包含loss和acc，如果指定验证集，还包含val_acc和val_loss，val_acc还需格外在model.compile启用metrics=['accuracy'])、on_batch_begin(logs包含size，即batch样本数)、on_batch_end(logs包含loss，若启用accuracy还有acc)
class LossHistory(kears.callbacks.Callback):
	def on_train_begin(self,logs={}):
		self.losses = []
	def on_batch_end(self,batch,logs={}): 
		self.losses.append(logs.get('loss'))

### tensorflow.keras.callbacks.TensorBard(log_dir='logs/',update_freq=32,..)
Tensorbard是一个可视化工具，而这个回调函数为Tensorboard编写一个日志，这样就可以可视化测试和训练的标准评估的动态图像
log_dir:用于保存被Tensorboard分析的日志文件的文件名，注意tf2会默认将log地址建立在程序所在文件，故可以直接写文件名，如果需要使用带'/'的地址，用'\\'代替(python写地址使用'/'但是tf2在windows系统下会报错)。
update_freq:当值为'batch'时，每个batch之后将损失和评估值写入到Tensorboard中，'epoch'同样，当是整数时，每整数个样本之后该回调函数会将损失和评估值写入到Tensorboard中，注意，频繁写入会减缓训练。

### tensorflow.keras.callbacks.ModelCheckpoint(modelcheckpoin_dir,monitor,save_best_only,mode)
用于存储训练好的网络模型及其权重到本地文件。
(1)monitor：表示要检测的loss值/指标值
	1.若是回归问题(预测一个连续值)，指标函数metrics(在model.compile中)设置的是MAE(keras.metrics.MeanAbsoluteError())的话，fit返回的history包含'mean_absolute_error'、'val_mean_absolute_error'，此时monitor可设为'mean_absolute_error'、'val_mean_absolute_error'
	2.若是二分类问题，指标函数metrics设置的是(keras.metrics.BinaryAccuracy())的话，fit返回的history包含'binary_accuracy'、'val_binary_accuracy'，此时monitor可设为'binary_accuracy'、'val_binary_accuracy'
	3.若是多分类问题，指标函数metrics设置的是(keras.metrics.CategoricalAccuracy())的话，fit返回的history包含'categorical_accuracy'、'val_categorical_accuracy'，此时monitor可设为'categorical_accuracy'、'val_categorical_accuracy'
(2)mode='min'/'max'/'auto':
	1.如果monitor设置的MAE等error指标，mode设为'min'，表示所要监视的error形式的指标越小越好
	2.如果monitor设置的accuracy指标，mode设为'max'，表示所要监视的accuracy形式的指标越小越好
	3.auto自动识别
(3)save_best_only:只储存最好的网络模型权重

### tensorflow.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode)
当被检测的数据不再提升，停止训练
monitor：被检测数据
min_delta：绝对变化小于该数据认为没有提升
patience：没有进步的训练轮数，在这之后训练被停止
mode:min模式被监测数据停止下降训练停止，max模式停止上升训练停止，auto模式方向会自动从被检测的数据的名字中判断出来

# utils.dataloader_medical.py
## keras.utils.Sequential()
用于构建数据生成器
用于整合数据集的基类，必须实现__getitem__(该方法应该包含一个完整的batch，作用是生成每个batch的数据)和__len__(计算每个epoch的迭代次数)方法，如果想在迭代之间修改数据集，可以通过定义on_epoch_end函数。
Sequence是进行多进程处理的更安全的方法，这种结构保证网络在每个epoch每个样本只训练一次，与生成器不同，但是定义的类的实例化对象可以用于model.fit_generator函数的生成器参数。
使用keras.utils.Sequential()处理的数据可以调用model.fit_generate里面的use_multiprocessing和workers，通过yield生成的数据则不行
例：class UnetDataset(keras.utils.Sequence): 
### np.random.rand()生成0~1的随机数
### math.ceil()返回大于或等于整数的最小整数(向上取整)

# utils.metrics.py
## tensorflow.kears.backend
keras可以基于两个backend，一个是Theano一个是Tensorflow，如果选择Tensorflow，keras就使用Tensorflow在底层搭建神经网络。
使用backend中的函数可以组成layer层，包含了很多数值处理方法。
使用Keras后端编写新代码(用keras后端API编写可以使Theano和Tensorflow都兼容)：
from keras import backend as K
inputs = K.variable([1,2]) 等价于tf.Variable()



