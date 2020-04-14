'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import time
import torch
import logging
import argparse
import torchvision
#from models import *
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
#from utils import progress_bar
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, type=bool, help='resume from checkpoint')
args = parser.parse_args()
logging.info(args)
#from rate import CyclicScheduler





store_name = "MC_ResNet50"
alpha = 0.0005
nb_epoch = 300
# setup output
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = store_name 








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

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((600,600)),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((600,600)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainset    = torchvision.datasets.ImageFolder(root='/data/Birds/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16, drop_last = True)


testset = torchvision.datasets.ImageFolder(root='/data/Birds/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=16, drop_last = True)

# Model


print('==> Building model..')




import torchvision.models as models

net = models.resnet50(pretrained=True)



groups = [10] * 152 + [11] * 48
groups = [0] +[sum(groups[:i+1])for i in range(len(groups))]


def Mask(nb_batch):
    foo  = [1] * 6 + [0] *  4
    foo2 = [1] * 7 + [0] *  4
    bar  = []
    bar2 = []
    for i in range(152):
        random.shuffle(foo)
        bar += foo
    for i in range(48):
        random.shuffle(foo2)
        bar += foo2
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,2048,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar



def supervisor(x,targets,height,cnum,num_classes):

        mask = Mask(x.size(0))

        features = x 

        features= [features[:, [range(groups[i-1],groups[i])] ] for i in range(1, len(groups))]


        features_1 = torch.cat(features[:num_classes], 2)
        features_2 = torch.cat(features[num_classes:], 2)
        features_1 = features_1.resize(features_1.size(0),features_1.size(2),features_1.size(3),features_1.size(4))
        features_2 = features_2.resize(features_2.size(0),features_2.size(2),features_2.size(3),features_2.size(4))

        features_1 = features_1.resize(features_1.size(0),features_1.size(1), features_1.size(2) * features_1.size(3))
        features_1 = F.softmax(features_1,2)
        features_1 = features_1.resize(features_1.size(0),features_1.size(1), x.size(2), x.size(2))
        features_1 = my_MaxPool2d(kernel_size=(1,cnum[0]), stride=(1,cnum[0]))(features_1)  
        features_1 = features_1.resize(features_1.size(0),features_1.size(1), features_1.size(2) * features_1.size(3))
        features_1_loss_2 = 1.0 - torch.mean(torch.sum(features_1,2))/(cnum[0] *1.0 ) # set margin = 2.0

        features_2 = features_2.resize(features_2.size(0),features_2.size(1), features_2.size(2) * features_2.size(3))
        features_2 = F.softmax(features_2,2)
        features_2 = features_2.resize(features_2.size(0),features_2.size(1), x.size(2), x.size(2))
        features_2 = my_MaxPool2d(kernel_size=(1,cnum[1]), stride=(1,cnum[1]))(features_2)  
        features_2 = features_2.resize(features_2.size(0),features_2.size(1), features_2.size(2) * features_2.size(3))
        features_2_loss_2 = 1.0 - torch.mean(torch.sum(features_2,2))/(cnum[1] *1.0 ) # set margin = 3.0


        loss_2 = features_1_loss_2 + features_2_loss_2


        features = x * mask

        features= [features[:, [range(groups[i-1],groups[i])] ] for i in range(1, len(groups))]


        features_1 = torch.cat(features[:num_classes], 2)
        features_2 = torch.cat(features[num_classes:], 2)
        features_1 = features_1.resize(features_1.size(0),features_1.size(2),features_1.size(3),features_1.size(4))
        features_2 = features_2.resize(features_2.size(0),features_2.size(2),features_2.size(3),features_2.size(4))
        old_features_1 = features_1
        old_features_2 = features_2
        #
        features_1_branch_1 = my_MaxPool2d(kernel_size=(1,cnum[0]), stride=(1,cnum[0]))(old_features_1)  
        features_1_branch_1 = nn.AvgPool2d(kernel_size=(height,height))(features_1_branch_1)
        features_1_branch_1 = features_1_branch_1.view(features_1_branch_1.size(0), -1)


        features_2_branch_1 = my_MaxPool2d(kernel_size=(1,cnum[1]), stride=(1,cnum[1]))(old_features_2)  
        features_2_branch_1 = nn.AvgPool2d(kernel_size=(height,height))(features_2_branch_1)
        features_2_branch_1 = features_2_branch_1.view(features_2_branch_1.size(0), -1)

        branch_1 = torch.cat([features_1_branch_1,features_2_branch_1], 1)


        loss_1 = criterion(branch_1, targets)
        
        return loss_1 + 10 * loss_2

class model_bn(nn.Module):
    def __init__(self, model, feature_size,classes_num):

        super(model_bn, self).__init__() 

        self.features = nn.Sequential(*list(net.children())[:-2])

        self.max = nn.MaxPool2d(kernel_size=14, stride=14)

        self.num_ftrs = 2048*1*1
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

        #pdb.set_trace()
        if self.training:
            branch_1_1oss = supervisor(x,targets,height=14,cnum=[10,11],num_classes=152)

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        loss = criterion(x, targets)
        
        #pdb.set_trace()
        if self.training:
            return x, loss, branch_1_1oss
        else:
            return x, loss



net =model_bn(net, 512, 200)


if use_cuda:
    net.classifier.cuda()
    net.features.cuda()

    net.classifier = torch.nn.DataParallel(net.classifier)
    net.features = torch.nn.DataParallel(net.features)

    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
# scheduler = CyclicScheduler(base_lr=0.00001, max_lr=0.01, step=2050., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle') ##exp_range ##triangular2


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #optimizer.param_groups[0]['lr'] = scheduler.get_rate()
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs, targets)


        loss   = outputs[1]
        branch_1_loss = outputs[2]

        loss = loss + alpha * branch_1_loss 

        loss.backward()
        optimizer.step()
        outputs = outputs[0]

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs,targets)
            
            #outputs = net(inputs[0].unsqueeze(0))
            loss = outputs[1]
            outputs = outputs[0] 

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100.*correct/total
    test_loss = test_loss/(idx+1)
    logging.info('test1, test_acc = %.4f,test_loss = %.4f' % (test_acc,test_loss))
    results_test_file.write('%d, %.4f,%.4f\n' % (epoch, test_acc,test_loss))
    results_test_file.flush()

    return test_acc
 


def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)



# optimizer = optim.SGD(net.classifier.parameters(), lr=0.0001, momentum=0.9, weight_decay=0)



optimizer = optim.SGD([
                        {'params': net.classifier.parameters(), 'lr': 0.1},
                        {'params': net.features.parameters(),   'lr': 0.01}
                        
                     ], 
                      momentum=0.9, weight_decay=5e-4)
max_val_acc = 0
lr = 0.1
for epoch in range(1, nb_epoch+1):
    optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch)
    optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch) / 10
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    train(epoch)
    val_acc = test(epoch)
    if val_acc >max_val_acc:
        max_val_acc = val_acc
        torch.save(net.state_dict(), store_name+'.pth')
    print("max_val_acc", max_val_acc)



