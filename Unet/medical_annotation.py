import random
import os

# in this case no validation set
trainval_percent = 1 
train_percent = 1
train_path = 'Mdeical_Datasets'

if __name__ == '__main__':
    random.seed(0)
    segfilepath = os.path.join(train_path,'Labels')
    savetxtpath = os.path.join(train_path,'ImageSets/Segmentation')

    tmp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in tmp_seg:
        if seg.endwith('.png'):
            total_seg.append(seg)
    num = len(total_seg)
    tv = int(num*trainval_percent)
    tr = int(num*train_percent)
    trainval = random.sample(range(total_seg),tv)
    train = random.sample(trainval,tr) # trainval = train+validation

    ftrainval = open(os.path.join(savetxtpath,'trainval.txt'),'w')
    ftest = open(os.path.join(savetxtpath,'test.txt'), 'w')  
    ftrain = open(os.path.join(savetxtpath,'train.txt'), 'w')  
    fval = open(os.path.join(savetxtpath,'val.txt'), 'w') 

    for i in range(num):
        name = total_seg[i][:-4]+'\n' # both are str
        if i in trainval:  # trainval+test = wholeSet
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name) 

    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
