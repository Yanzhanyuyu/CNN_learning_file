import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
import torch.nn.functional as F 
import torch.optim as optim
batch_size = 8#每次拿取的数据量
transform = transforms.Compose([
    #转为张量
    transforms.ToTensor(),
    #归一化
    transforms.Normalize((0.1307, ), (0.3081, ))
])


#dataset和dataloader
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform
                          )# PyTorch 中的 MNIST 数据集
#测试dataset

#DataLoader
#测试DataLoader是否有效
# dataiter = iter(train_loader)#iter()是Python内置函数，用于返回一个迭代器对象。
#以用于遍历train_loader中的数据。
#展示batch的数据
# images, labels = dataiter.__next__()
# print(images)
# print(labels)


#训练集的dataset和dataloader
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform
                               )
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size
                          )
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=False,
                               download=True,
                               transform=transform
                               )
test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size
                          )





# CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层#输入通道数，输入数据是灰度图像，所以通道数为1。
        #输出通道数（out_channels）是指卷积层输出的特征图的通道数。第一个卷积层输出10个特征图
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        #池化层，减小特征的尺寸，减小计算量，同时防止过拟合
        self.pooling = torch.nn.MaxPool2d(2) 
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        return x
model = Net()
#将模型放入GPU中
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #放入cuda中
#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
#设定冲量，使得模型更好的收敛
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

def test():
    correct = 0
    total = 0
    with torch.no_grad():  #不需要计算梯度
        for data in test_loader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            #在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#得到预测输出
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('精确度:%d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()