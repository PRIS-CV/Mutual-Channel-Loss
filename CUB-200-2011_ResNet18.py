'''PyTorch CUB-200-2011 Training with ResNet18 (TRAINED FROM SCRATCH).
   NOTICE: for baseline, the channel of the final features should keep same with the Vanilla ResNet18'''
from __future__ import print_function
import os
# import nni
import time
import torch
import logging
import argparse
import torchvision
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from my_pooling import my_MaxPool2d,my_AvgPool2d
import torchvision.transforms as transforms


logger = logging.getLogger('MC_ResNet18_224')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr = 0.1
nb_epoch = 300
criterion = nn.CrossEntropyLoss()

#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainset    = torchvision.datasets.ImageFolder(root='/home/data/Birds/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=16, drop_last = True)

testset = torchvision.datasets.ImageFolder(root='/home/data/Birds/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=16)



print('==> Building model..')

# Model

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 600, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


net = resnet18(pretrained=False)



def Mask(nb_batch, channels):

    foo = [1] * 2 + [0] *  1
    bar = []
    for i in range(200):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,200*channels,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

def supervisor(x,targets,height,cnum):
        mask = Mask(x.size(0), cnum)
        branch = x
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        branch = F.softmax(branch,2)
        branch = branch.reshape(branch.size(0),branch.size(1), x.size(2), x.size(2))
        branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)  
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0

        branch_1 = x * mask 

        branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)  
        branch_1 = nn.AvgPool2d(kernel_size=(height,height))(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)

        loss_1 = criterion(branch_1, targets)
        
        return [loss_1, loss_2] 

class model_bn(nn.Module):
    def __init__(self, feature_size=512,classes_num=200):

        super(model_bn, self).__init__() 

        self.features = nn.Sequential(*list(net.children())[:-2]) 

        self.max = nn.MaxPool2d(kernel_size=14, stride=14)

        self.num_ftrs = 600*1*1
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, targets):



        x = self.features(x)

        if self.training:
            MC_loss = supervisor(x,targets,height=14,cnum=3)

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        loss = criterion(x, targets)

        if self.training:
            return x, loss, MC_loss
        else:
            return x, loss


use_cuda = torch.cuda.is_available()



net =model_bn(512, 200)

if use_cuda:
    net.classifier.cuda()
    net.features.cuda()


    net.classifier = torch.nn.DataParallel(net.classifier)
    net.features = torch.nn.DataParallel(net.features)


    cudnn.benchmark = True


def train(epoch,net, args, trainloader,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out, ce_loss, MC_loss = net(inputs, targets)

        loss = ce_loss + args["alpha_1"] * MC_loss[0] +   args["beta_1"]  * MC_loss[1] 

        loss.backward()
        optimizer.step()


        train_loss += loss.item()

        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()



    train_acc = 100.*correct/total
    train_loss = train_loss/(idx+1)
    logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc,train_loss))
    return train_acc, train_loss

def test(epoch,net,testloader,optimizer):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            out, ce_loss = net(inputs,targets)
            
            test_loss += ce_loss.item()
            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()


    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('test, test_acc = %.4f,test_loss = %.4f' % (test_acc,test_loss))

    return test_acc
 
def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)


optimizer = optim.SGD([
                        {'params': net.classifier.parameters(), 'lr': 0.1},
                        {'params': net.features.parameters(),   'lr': 0.1},
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4)


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MC2_AutoML Example')

    parser.add_argument('--alpha_1', type=float, default=1.5, metavar='ALPHA',
                        help='alpha_1 value (default: 2.0)')
    parser.add_argument('--beta_1', type=float, default=20.0, metavar='BETA',
                        help='beta_1 value (default: 20.0)')

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    try:
        args = vars(get_params())
        print(args)
        # main(params)
        max_val_acc = 0
        for epoch in range(1, nb_epoch+1):
            if epoch ==150:
                lr = 0.01
            if epoch ==225:
                lr = 0.001
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr 

            train(epoch, net, args,trainloader,optimizer)
            test_acc = test(epoch, net,testloader,optimizer)
            if test_acc >max_val_acc:
                max_val_acc = test_acc

            print("max_val_acc", max_val_acc)


    except Exception as exception:
        logger.exception(exception)
        raise

