import torch
import os
from torchvision import transforms
import My_dataset
from torch.utils.data import DataLoader
from CNN import CNN
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter(log_dir='runs/flowers_experiment')
USE_GPU = True
LR = 0.0001
TIMES = 20
batch_size = 8
num_worker = min([os.cpu_count(), batch_size if batch_size>1 else 0,8]) # type: ignore
def main(root:str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_path, train_label, test_path, test_label = My_dataset.read_split(root=root ,test_rate=0.1)

    data_transforms = {                      #数据集的处理方法
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test":transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    #训练集
    train_set = My_dataset.My_Dataset(img_path=train_path,
                           img_label=train_label,
                           transforms=data_transforms["train"])
    #测试集
    test_set = My_dataset.My_Dataset(img_path=test_path,
                          img_label=test_label,
                          transforms=data_transforms["test"])
    
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_worker,
                              collate_fn=train_set.collate_fn)
    test_loader = DataLoader(dataset=test_set,
                             shuffle=True,
                             collate_fn=train_set.collate_fn)


    #My_dataset.plot_load_image(train_loader)

    #test 无需打包
    cnn = CNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    if torch.cuda.is_available() and USE_GPU == True:
        cnn = cnn.cuda()
        loss_function = loss_function.cuda()
    
    #train
    for times in range(TIMES):
        for data in train_loader:
            images, labels = data
            print(images.shape)
            exit()
            if torch.cuda.is_available() and USE_GPU == True:
                images = images.cuda()
                labels = labels.cuda()
            output = cnn(images)
            loss = loss_function(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('训练损失值', loss, times)
        writer.add_scalar('梯度', optimizer.param_groups[0]["lr"], times)
    
    '''
    #test
    wrong_num = 0
    for data in test_loader:
        x, real_y = data
        if torch.cuda.is_available() and USE_GPU == True:
            x = x.cuda()
            real_y = real_y.cuda()
        pred_y = cnn(x)
        if pred_y != real_y:
            wrong_num+=1
    print("the right num:{}".format(int(len(train_loader))))
    '''

    #test
    wrong_num = 0
    i = 0
    for data in test_loader:
        images, labels = data
        if torch.cuda.is_available() and USE_GPU == True:
            images = images.cuda()
            labels = labels.cuda()
        #print(images)
        #exit()
        tmp_output = cnn(images)
        pred_y = torch.max(torch.softmax(tmp_output, dim=1), dim=1)[1].data
        if torch.cuda.is_available() and USE_GPU == True:
            pred_y = pred_y.cuda()
        if pred_y != labels:
            print(pred_y, labels)
            wrong_num +=1
        i+=1
    print("wrong num:{} , sum:{}".format(wrong_num, i))

a = './flower_photos'

if __name__ == '__main__':
    main(a)
