import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,5,1,2)
        self.pool1 = nn.MaxPool2d(8)
        self.conv2 = nn.Conv2d(16,32,5,1,2)
        self.pool2 = nn.MaxPool2d(4)
        self.fc = nn.Linear(32*7*7,5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        tmp = x.shape
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        tmp = x.shape
        x = x.view(-1, 32*7*7)
        x= self.fc(x)
        return x


