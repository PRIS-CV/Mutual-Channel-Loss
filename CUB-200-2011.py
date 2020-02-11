'''PyTorch CUB-200-2011 Training without pre_trained model.'''
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch CUB-200-2011 Training without pre_trained model')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, type=bool, help='resume from checkpoint')
args = parser.parse_args()
logging.info(args)



store_name = "CUB-200-2011"
alpha = 1.5
should_mask = True
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = store_name 





nb_epoch = 300



try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)
logging.info("OPENING " + exp_dir + '/results_train.csv')
logging.info("OPENING " + exp_dir + '/results_test.csv')


results_train_file = open(exp_dir + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc,train_loss\n')
results_train_file.flush()

results_test_file = open(exp_dir + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc,test_loss\n')
results_test_file.flush()



use_cuda = torch.cuda.is_available()

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




trainset    = torchvision.datasets.ImageFolder(root='/home/changdongliang/data/Birds2/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)



testset = torchvision.datasets.ImageFolder(root='/home/changdongliang/data/Birds2/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)


print('==> Building model..')

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 600],
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



import torchvision.models as models

net = VGG('VGG16')
net = net.features



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
        branch = branch.resize(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        branch = F.softmax(branch,2)
        branch = branch.resize(branch.size(0),branch.size(1), x.size(2), x.size(2))
        branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)  
        branch = branch.resize(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
        loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0

        if should_mask==True:
            branch_1 = x * mask 
        else:
            branch_1 = x
        branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)  
        branch_1 = nn.AvgPool2d(kernel_size=(height,height))(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)

        loss_1 = criterion(branch_1, targets)
        
        return loss_1  + 10 * loss_2 

class model_bn(nn.Module):
    def __init__(self, model, feature_size,classes_num):

        super(model_bn, self).__init__() 

        self.features = net

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



net =model_bn(net, 512, 200)



if use_cuda:
    net.cuda()
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs, targets)


        loss   = outputs[1]
        MC_loss = outputs[2]

        loss = loss + alpha * MC_loss 

        loss.backward()
        optimizer.step()
        outputs = outputs[0]

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()



    train_acc = 100.*correct/total
    train_loss = train_loss/(idx+1)
    logging.info('Iteration %d, train_acc = %.5f,train_loss = %.6f' % (epoch, train_acc,train_loss))
    results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc,train_loss))
    results_train_file.flush()
    return train_acc, train_loss

def test(epoch):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs,targets)
        
        loss = outputs[1]
        outputs = outputs[0] 

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()


    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('test, test_acc = %.4f,test_loss = %.4f' % (test_acc,test_loss))
    results_test_file.write('%d, %.4f,%.4f\n' % (epoch, test_acc,test_loss))
    results_test_file.flush()

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




max_val_acc = 0
lr = 0.1
for epoch in range(1, nb_epoch+1):
    if epoch ==150:
        lr = 0.01
    if epoch ==225:
        lr = 0.001
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr 
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train(epoch)
    val_acc = test(epoch)
    if val_acc >max_val_acc:
        max_val_acc = val_acc
        torch.save(net.state_dict(), store_name+'.pth')


print(max_val_acc)


