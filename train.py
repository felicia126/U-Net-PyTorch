import os
import time
import networks
import numpy as np
from subprocess import call
from loss import dice as dice_loss
from data_load import CarDataSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


### variables ###

model_name = 'DUNet'

augment = False
dropout = False

# using dice loss or cross-entropy loss
dice = True

# learning rate, batch size, samples per epoch, epoch where to lower learning rate and total number of epochs
lr = 1e-3
batch_size = 1
num_samples = 100
low_lr_epoch = 50
epochs = 100

#################

image_directory = 'data/train'
mask_directory = 'data/train_masks'

print(model_name)
print("augment="+str(augment)+" dropout="+str(dropout))
print(str(epochs) + " epochs - lr: " + str(lr) + " - batch size: " + str(batch_size))

# GPU enabled
cuda = torch.cuda.is_available()

# network and optimizer
net = networks.UNetSmall()#DenseUNet(input_features=3, network_depth=4, block_length=4, num_init_features=16, growth_rate=4, bn_size=4, drop_rate=0):
if cuda: net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)

# data loader
cars = CarDataSet(image_directory=image_directory, mask_directory=mask_directory, augment=augment)

val_idx = np.random.choice(range(cars.__len__()), 200, replace=False)
train_idx = [i for i in list(range(cars.__len__())) if i not in val_idx]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

train_data = torch.utils.data.DataLoader(cars, batch_size=batch_size, shuffle=True, sampler=train_sampler, num_workers=0)
val_data = torch.utils.data.DataLoader(cars, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=0)

print len(train_idx)

# train loop

print('Start training...')

for epoch in range(epochs):

    running_loss = 0.0

    # lower learning rate
    if epoch == low_lr_epoch:
        for param_group in optimizer.param_groups:
            lr = lr / 10
            param_group['lr'] = lr

    # switch to train mode
    net.train()
    
    start = time.time()
    for i, data in enumerate(train_data):

        # wrap data in Variables
        inputs, labels = data
        if cuda: inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        # forward pass and loss calculation
        outputs = net(inputs)

        # get either dice loss or cross-entropy
        if dice:
            outputs = outputs[:,1,:,:].unsqueeze(dim=1)
            loss = dice_loss(outputs, labels)
        else:
            labels = labels.squeeze(dim=1)
            loss = criterion(outputs, labels)

        # empty gradients, perform backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # save and print statistics
        running_loss += loss.data[0]

        if (i+1)%100 == 0: break
    
    # print statistics
    if dice:
        print('  [epoch %d] - train dice loss: %.3f' % (epoch + 1, running_loss/(i+1)))
    else:
        print('  [epoch %d] - train cross-entropy loss: %.3f' % (epoch + 1, running_loss/(i+1)))

    # only validate every 10 epochs
    #if (epoch+1)%10 != 0: continue
    
    # switch to eval mode
    net.eval()

    all_dice = []
    all_accuracy = []

    for i, data in enumerate(val_data):

        # wrap data in Variable
        inputs, labels = data
        if cuda: inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        
        # inference
        outputs = net(inputs)
        
        # log softmax into softmax
        if not dice: outputs = outputs.exp()

        # round outputs to either 0 or 1
        outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()

        # accuracy
        outputs, labels = outputs.data.cpu().numpy(), labels.data.cpu().numpy()
        accuracy = (outputs == labels).sum() / float(outputs.size)

        # dice
        intersect = (outputs+labels==2).sum()
        union = np.sum(outputs) + np.sum(labels)
        dice = 1 - (2 * intersect + 1e-5) / (union + 1e-5)

        all_dice.append(dice)
        all_accuracy.append(accuracy)

    print('    val dice loss: %.9f - val accuracy: %.8f' % (np.mean(all_dice), np.mean(all_accuracy)))
    
# save weights

torch.save(net, "model_"+str(model_name)+".pht")

print('Finished training...')
