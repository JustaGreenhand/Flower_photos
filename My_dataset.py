from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import random
import numpy
import matplotlib.pyplot as plt

class My_Dataset(Dataset):
    def __init__(self, img_path: list, img_label: list, transforms= None):
        self.img_path = img_path
        self.img_label = img_label
        self.transforms = transforms

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, item):
        img = Image.open(self.img_path[item])
        if img.mode != 'RGB':    #只处理RGB图像
            raise ValueError("image:{} is not the RGB".format(self.img_path[item]))
        label = self.img_label[item]

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


#test_rate  测试集占全部数据的百分比。默认是0.2
def read_split(root: str, test_rate: float = 0.2):
    if os.path.exists(root) == False:
        print("--the dataset does not exict.--")
        exit()
    #这里默认是遍历文件并提取出文件夹，也就是类别的名称
    Myclass=[cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    #print(Myclass)
    Myclass.sort()
    #建立索引
    index = list(range(0,len(Myclass)))
    Myclass_index = {Myclass[i]: index[i] for i in range(len(Myclass))}
    #print(Myclass_index)
    #print(Myclass_index['roses'])
    file = open('./index.index','w')
    file.write(str(Myclass_index))
    file.close()

    train_path = []
    train_label = []

    test_path = []
    test_label = []
    class_num = []   #每个类别的样本个数
    
    for cla in Myclass:
        cla_path = os.path.join(root, cla)    #类别的文件目录’
        img_path = [os.path.join(root, cla, name) for name in os.listdir(cla_path)]
        img_class = Myclass_index[cla]   #记录图片所属的类别
        #print(img_path)
        class_num.append(len(img_path))

        test_path_tmp = random.sample(img_path, k=int(len(img_path)*test_rate))
        
        for path in img_path:
            if path in test_path_tmp:
                test_path.append(path)
                test_label.append(img_class)
            else:
                train_path.append(path)
                train_label.append(img_class)
    

    return train_path, train_label, test_path, test_label



def plot_load_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    path = './index.index'
    assert os.path.exists(path), path + 'does not exist!!'
    file = open(path ,'r')
    class_index = eval(file.read())
    class_index = dict(zip(class_index.values(), class_index.keys()))
    print(class_index)
    for data in data_loader:
        images, tmp = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1,2,0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            
            labels = tmp[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_index[labels])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()