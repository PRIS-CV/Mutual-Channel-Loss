'''PyTorch CUB-200-2011 Training with VGG16 (TRAINED FROM SCRATCH).'''
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


logger = logging.getLogger('MC_VGG_224')


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 600, 'M', 512, 512, 600],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)








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

        self.features_1 = nn.Sequential(*list(VGG('VGG16').features.children())[:34])
        self.features_2 = nn.Sequential(*list(VGG('VGG16').features.children())[34:])

        self.max = nn.MaxPool2d(kernel_size=2, stride=2)

        self.num_ftrs = 600*7*7
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


        x = self.features_1(x)

        x = self.features_2(x)

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
    net.features_1.cuda()
    net.features_2.cuda()

    net.classifier = torch.nn.DataParallel(net.classifier)
    net.features_1 = torch.nn.DataParallel(net.features_1)
    net.features_2 = torch.nn.DataParallel(net.features_2)

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
                        {'params': net.features_1.parameters(),   'lr': 0.1},
                        {'params': net.features_2.parameters(),   'lr': 0.1},
                        
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
            optimizer.param_groups[2]['lr'] = lr 

            train(epoch, net, args,trainloader,optimizer)
            test_acc = test(epoch, net,testloader,optimizer)
            if test_acc >max_val_acc:
                max_val_acc = test_acc

            print("max_val_acc", max_val_acc)


    except Exception as exception:
        logger.exception(exception)
        raise

