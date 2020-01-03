import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("CPU")

# 超参数
epoch = 1
batch_size = 32
lr = 0.01

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #print("1", x.size())
        x = x.view(x.size()[0], -1)
        #print("2", x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #模型保存路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  #模型加载路径
opt = parser.parse_args()

# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]))
# val_dataset = datasets.ImageFolder(
#     valdir,
#     transforms.Compose([
#     transforms.ToTensor(),
# ]))
# train_sampler = None
# val_sampler = None
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=2, shuffle=(train_sampler is None),
#     num_workers=1, pin_memory=(train_sampler is None), sampler=train_sampler)
#
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False,
#                                          num_workers=1, pin_memory=True, sampler=val_sampler)
# 数据预处理，一定要有totensor
transform = transforms.ToTensor()

train_dataset = tv.datasets.MNIST(
    root = './data/',
    train = True,
    download = True,
    transform = transform
)

test_dataset = tv.datasets.MNIST(
    root= './data/',
    train = False,
    download = True,
    transform = transform
)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True
)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
)

def main():
    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimzizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
    for e in range(epoch):
        sum_loss = 0.0

        for i,data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            # inputs = torch.cuda.FloatTensor(input.cuda())
            # targets= torch.cuda.LongTensor(target.cuda())

            optimzizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimzizer.step()

            #print loss on 100th epoch
            sum_loss += loss.item()
            if i % 100 == 99:
                print("memory_allocated:%f Mb" % float(torch.cuda.memory_allocated() / 1024 ** 2))
                print("max_memory_allocated:%f Mb" % float(torch.cuda.max_memory_allocated() / 1024 ** 2))
                # print("memory_cached:%f Mb" % float(torch.cuda.memory_cached() / 1024 ** 2))
                print('[%d, %d] loss: %.03f'
                      % (e + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

        #with torch.no_grad():
        # correct = 0
        # total = 0
        # for data in testloader:
        #     images, labels = data
        #     images, labels = images.to(device), labels.to(device)
        #     outputs = net(images)
        #     print("memory_allocated:%f Mb" % float(torch.cuda.memory_allocated()/1024**2))
        #     _,pre = torch.max(outputs.data,1)
        #     total += labels.size(0)
        #     correct += (pre == labels).sum()
        # print('第%d个epoch的识别准确率为：%d%%' % (e + 1, (100 * correct / total)))
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            print(torch.cuda.is_available())
            print("memory_allocated:%f Mb" % float(torch.cuda.memory_allocated()/1024**2))
            print("max_memory_allocated:%f Mb" % float(torch.cuda.max_memory_allocated() / 1024 ** 2))
            print("memory_cached:%f Mb" % float(torch.cuda.memory_cached()/1024**2))

            _,pre = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (pre == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (e + 1, (100 * correct / total)))

if __name__== '__main__':
    time_s = time.time()
    main()
    time_e = time.time()
    print("time cost : ",time_e-time_s)