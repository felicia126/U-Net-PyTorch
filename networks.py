import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# implementation of U-Net as described in the paper
# & padding to keep input and output sizes the same

class UNet(nn.Module):

    def __init__(self, dice=False):

        super(UNet, self).__init__()

        self.conv1_input =      nn.Conv2d(1, 64, 3, padding=1)
        self.conv1 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_input =      nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_input =      nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_input =      nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_input =      nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 =            nn.Conv2d(1024, 1024, 3, padding=1)

        self.conv6_up =         nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_input =      nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv7_up =         nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_input =      nn.Conv2d(512, 256, 3, padding=1)
        self.conv7 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv8_up =         nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_input =      nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv9_up =         nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_input =      nn.Conv2d(128, 64, 3, padding=1)
        self.conv9 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_output =     nn.Conv2d(64, 2, 1)
        
        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))

        layer6 = F.relu(self.conv6_up(layer5))
        layer6 = torch.cat((layer4, layer6), 1)
        layer6 = F.relu(self.conv6_input(layer6))
        layer6 = F.relu(self.conv6(layer6))

        layer7 = F.relu(self.conv7_up(layer6))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
        layer9 = self.final(self.conv9_output(layer9))

        return layer9


# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling

class VNet(nn.Module):

    def __init__(self, dice=False):

        super(VNet, self).__init__()

        self.conv1 =        nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv1 =        nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv1_down =   nn.Conv2d(16, 32, 2, stride=2, padding=0)

        self.conv2a =       nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2b =       nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2_down =   nn.Conv2d(32, 64, 2, stride=2, padding=0)

        self.conv3a =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3b =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3c =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv3_down =   nn.Conv2d(64, 128, 2, stride=2, padding=0)

        self.conv4a =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4b =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4c =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv4_down =   nn.Conv2d(128, 256, 2, stride=2, padding=0)

        self.conv5a =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5b =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5c =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv5_up =     nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)

        self.conv6a =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6b =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6c =       nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv6_up =     nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0)

        self.conv7a =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7b =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7c =       nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.conv7_up =     nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)

        self.conv8a =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv8b =       nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv8_up =     nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0)

        self.conv9 =        nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv9_1x1 =    nn.Conv2d(32, 2, 1, stride=1, padding=0)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.conv1(x))
        layer1 = torch.add(layer1, torch.cat([x]*16,1))

        conv1 = F.relu(self.conv1_down(layer1))

        layer2 = F.relu(self.conv2a(conv1))
        layer2 = F.relu(self.conv2b(layer2))
        layer2 = torch.add(layer2, conv1)

        conv2 = F.relu(self.conv2_down(layer2))

        layer3 = F.relu(self.conv3a(conv2))
        layer3 = F.relu(self.conv3b(layer3))
        layer3 = F.relu(self.conv3c(layer3))
        layer3 = torch.add(layer3, conv2)

        conv3 = F.relu(self.conv3_down(layer3))

        layer4 = F.relu(self.conv4a(conv3))
        layer4 = F.relu(self.conv4b(layer4))
        layer4 = F.relu(self.conv4c(layer4))
        layer4 = torch.add(layer4, conv3)

        conv4 = F.relu(self.conv4_down(layer4))

        layer5 = F.relu(self.conv5a(conv4))
        layer5 = F.relu(self.conv5b(layer5))
        layer5 = F.relu(self.conv5c(layer5))
        layer5 = torch.add(layer5, conv4)

        conv5 = F.relu(self.conv5_up(layer5))

        cat6 = torch.cat((conv5, layer4), 1)

        layer6 = F.relu(self.conv6a(cat6))
        layer6 = F.relu(self.conv6b(layer6))
        layer6 = F.relu(self.conv6c(layer6))
        layer6 = torch.add(layer6, cat6)

        conv6 = F.relu(self.conv6_up(layer6))

        cat7 = torch.cat((conv6, layer3), 1)

        layer7 = F.relu(self.conv7a(cat7))
        layer7 = F.relu(self.conv7b(layer7))
        layer7 = F.relu(self.conv7c(layer7))
        layer7 = torch.add(layer7, cat7)

        conv7 = F.relu(self.conv7_up(layer7))

        cat8 = torch.cat((conv7, layer2), 1)

        layer8 = F.relu(self.conv8a(cat8))
        layer8 = F.relu(self.conv8b(layer8))
        layer8 = torch.add(layer8, cat8)

        conv8 = F.relu(self.conv8_up(layer8))

        cat9 = torch.cat((conv8, layer1), 1)

        layer9 = F.relu(self.conv9(cat9))
        layer9 = torch.add(layer9, cat9)
        layer9 = self.final(self.conv9_1x1(layer9))

        return layer9


# 2D variation of VNet - similar to UNet
# added residual functions to each block
# & down convolutions instead of pooling
# & batch normalization for convolutions
# & drop out before every upsample layer

class VNet_Xtra(nn.Module):

    def __init__(self, dice=True, dropout=False, input_features=3):

        super(VNet_Xtra, self).__init__()

        self.dropout = dropout
        if self.dropout:
            self.do6 = nn.Dropout2d()
            self.do7 = nn.Dropout2d()
            self.do8 = nn.Dropout2d()
            self.do9 = nn.Dropout2d()

        self.conv1 =        nn.Conv2d(input_features, 16, 3, stride=1, padding=1)
        self.bn1 =          nn.BatchNorm2d(16)
        self.conv1_down =   nn.Conv2d(16, 32, 2, stride=2, padding=0)
        self.bn1_down =     nn.BatchNorm2d(32)

        self.conv2a =       nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2a =         nn.BatchNorm2d(32)
        self.conv2b =       nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2b =         nn.BatchNorm2d(32)
        self.conv2_down =   nn.Conv2d(32, 64, 2, stride=2, padding=0)
        self.bn2_down =     nn.BatchNorm2d(64)

        self.conv3a =       nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3a =         nn.BatchNorm2d(64)
        self.conv3b =       nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3b =         nn.BatchNorm2d(64)
        self.conv3c =       nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3c =         nn.BatchNorm2d(64)
        self.conv3_down =   nn.Conv2d(64, 128, 2, stride=2, padding=0)
        self.bn3_down =     nn.BatchNorm2d(128)

        self.conv4a =       nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4a =         nn.BatchNorm2d(128)
        self.conv4b =       nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4b =         nn.BatchNorm2d(128)
        self.conv4c =       nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4c =         nn.BatchNorm2d(128)
        self.conv4_down =   nn.Conv2d(128, 256, 2, stride=2, padding=0)
        self.bn4_down =     nn.BatchNorm2d(256)

        self.conv5a =       nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn5a =         nn.BatchNorm2d(256)
        self.conv5b =       nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn5b =         nn.BatchNorm2d(256)
        self.conv5c =       nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn5c =         nn.BatchNorm2d(256)
        self.conv5_up =     nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.bn5_up =       nn.BatchNorm2d(128)

        self.conv6a =       nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn6a =         nn.BatchNorm2d(256)
        self.conv6b =       nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn6b =         nn.BatchNorm2d(256)
        self.conv6c =       nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn6c =         nn.BatchNorm2d(256)
        self.conv6_up =     nn.ConvTranspose2d(256, 64, 2, stride=2, padding=0)
        self.bn6_up =       nn.BatchNorm2d(64)

        self.conv7a =       nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn7a =         nn.BatchNorm2d(128)
        self.conv7b =       nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn7b =         nn.BatchNorm2d(128)
        self.conv7c =       nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn7c =         nn.BatchNorm2d(128)
        self.conv7_up =     nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)
        self.bn7_up =       nn.BatchNorm2d(32)

        self.conv8a =       nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn8a =         nn.BatchNorm2d(64)
        self.conv8b =       nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn8b =         nn.BatchNorm2d(64)
        self.conv8_up =     nn.ConvTranspose2d(64, 16, 2, stride=2, padding=0)
        self.bn8_up =       nn.BatchNorm2d(16)

        self.conv9 =        nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn9 =          nn.BatchNorm2d(32)
        self.conv9_1x1 =    nn.Conv2d(32, 2, 1, stride=1, padding=0)
        self.bn9_1x1 =      nn.BatchNorm2d(2)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        layer1 = F.relu(self.bn1(self.conv1(x)))

        conv1 = F.relu(self.bn1_down(self.conv1_down(layer1)))

        layer2 = F.relu(self.bn2a(self.conv2a(conv1)))
        layer2 = F.relu(self.bn2b(self.conv2b(layer2)))
        layer2 = torch.add(layer2, conv1)

        conv2 = F.relu(self.bn2_down(self.conv2_down(layer2)))

        layer3 = F.relu(self.bn3a(self.conv3a(conv2)))
        layer3 = F.relu(self.bn3b(self.conv3b(layer3)))
        layer3 = F.relu(self.bn3c(self.conv3c(layer3)))
        layer3 = torch.add(layer3, conv2)

        conv3 = F.relu(self.bn3_down(self.conv3_down(layer3)))

        layer4 = F.relu(self.bn4a(self.conv4a(conv3)))
        layer4 = F.relu(self.bn4b(self.conv4b(layer4)))
        layer4 = F.relu(self.bn4c(self.conv4c(layer4)))
        layer4 = torch.add(layer4, conv3)

        conv4 = F.relu(self.bn4_down(self.conv4_down(layer4)))

        layer5 = F.relu(self.bn5a(self.conv5a(conv4)))
        layer5 = F.relu(self.bn5b(self.conv5b(layer5)))
        layer5 = F.relu(self.bn5c(self.conv5c(layer5)))
        layer5 = torch.add(layer5, conv4)

        conv5 = F.relu(self.bn5_up(self.conv5_up(layer5)))

        cat6 = torch.cat((conv5, layer4), 1)

        if self.dropout: cat6 = self.do6(cat6)

        layer6 = F.relu(self.bn6a(self.conv6a(cat6)))
        layer6 = F.relu(self.bn6b(self.conv6b(layer6)))
        layer6 = F.relu(self.bn6c(self.conv6c(layer6)))
        layer6 = torch.add(layer6, cat6)

        conv6 = F.relu(self.bn6_up(self.conv6_up(layer6)))

        cat7 = torch.cat((conv6, layer3), 1)

        if self.dropout: cat7 = self.do7(cat7)

        layer7 = F.relu(self.bn7a(self.conv7a(cat7)))
        layer7 = F.relu(self.bn7b(self.conv7b(layer7)))
        layer7 = F.relu(self.bn7c(self.conv7c(layer7)))
        layer7 = torch.add(layer7, cat7)

        conv7 = F.relu(self.bn7_up(self.conv7_up(layer7)))

        cat8 = torch.cat((conv7, layer2), 1)

        if self.dropout: cat8 = self.do8(cat8)

        layer8 = F.relu(self.bn8a(self.conv8a(cat8)))
        layer8 = F.relu(self.bn8b(self.conv8b(layer8)))
        layer8 = torch.add(layer8, cat8)

        conv8 = F.relu(self.bn8_up(self.conv8_up(layer8)))

        cat9 = torch.cat((conv8, layer1), 1)

        if self.dropout: cat9 = self.do9(cat9)

        layer9 = F.relu(self.bn9(self.conv9(cat9)))
        layer9 = torch.add(layer9, cat9)
        layer9 = self.final(self.bn9_1x1(self.conv9_1x1(layer9)))

        return layer9


# a smaller version of UNet
# used for testing purposes

class UNetSmall(nn.Module):

    def __init__(self, dice=True):

        super(UNetSmall, self).__init__()

        self.conv1_input =      nn.Conv2d(3, 8, 3, padding=1)
        self.conv1 =            nn.Conv2d(8, 8, 3, padding=1)
        self.conv2_input =      nn.Conv2d(8, 32, 3, padding=1)
        self.conv2 =            nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_input =      nn.Conv2d(32, 128, 3, padding=1)
        self.conv3 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_input =      nn.Conv2d(128, 512, 3, padding=1)
        self.conv4 =            nn.Conv2d(512, 512, 3, padding=1)

        self.conv7_up =         nn.ConvTranspose2d(512, 128, 2, 2)
        self.conv7_input =      nn.Conv2d(128+128, 128, 3, padding=1)
        self.conv7 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv8_up =         nn.ConvTranspose2d(128, 32, 2, 2)
        self.conv8_input =      nn.Conv2d(32+32, 32, 3, padding=1)
        self.conv8 =            nn.Conv2d(32, 32, 3, padding=1)
        self.conv9_up =         nn.ConvTranspose2d(32, 8, 2, 2)
        self.conv9_input =      nn.Conv2d(8+8, 8, 3, padding=1)
        self.conv9 =            nn.Conv2d(8, 8, 3, padding=1)
        self.conv9_output =     nn.Conv2d(8, 2, 1)

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def switch(self, dice):

        if dice:
            self.final =        F.softmax
        else:
            self.final =        F.log_softmax

    def forward(self, x):

        #print "start"

        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        #print "layer1: " + str(layer1.size())

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        #print "layer2: " + str(layer2.size())

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        #print "layer3: " + str(layer3.size())

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        #print "layer4: " + str(layer4.size())

        layer7 = F.relu(self.conv7_up(layer4))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        #print "layer7: " + str(layer7.size())

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        #print "layer8: " + str(layer8.size())

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
        layer9 = self.final(self.conv9_output(layer9))

        #print "end"

        return layer9

### DENSE U NET

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _DownSample(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_DownSample, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=2, stride=2, bias=False))

class _UpSample(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpSample, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.ConvTranspose2d(num_input_features, num_output_features,
                                          kernel_size=2, stride=2, bias=False))

class DenseUNet(nn.Module):

    def __init__(self, input_features=3, network_depth=4, block_length=4, num_init_features=16, growth_rate=4, bn_size=4, drop_rate=0):
        super(DenseUNet, self).__init__()

        # Input

        self.conv0 = nn.Conv2d(input_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        #self.down0 = transition = _DownSample(num_input_features=num_init_features, num_output_features=num_init_features)
        #self.pool0 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        num_features = num_init_features

        # Encoder

        skip_connections = []
        self.encoder_blocks = []
        self.encoder_sample = []
        for i in range(network_depth-1):
            denseblock = _DenseBlock(num_layers=block_length, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            num_features = num_features + block_length * growth_rate
            skip_connections.append(num_features)
            self.encoder_blocks.append(denseblock)
            transition = _DownSample(num_input_features=num_features, num_output_features=num_features)
            num_features = num_features
            self.encoder_sample.append(transition)
            growth_rate = growth_rate * 2
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.encoder_sample = nn.ModuleList(self.encoder_sample)

        # Bottom
        self.bottom_block = _DenseBlock(num_layers=block_length, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_length * growth_rate
        # Decoder

        self.decoder_blocks = []
        self.decoder_sample = []
        for i, skip in zip(range(network_depth-1), skip_connections[::-1]):
            growth_rate = growth_rate // 2
            div = 4 if i==0 else 8
            transition = _UpSample(num_input_features=num_features, num_output_features=num_features//div)
            num_features = num_features // div + skip
            self.decoder_sample.append(transition)
            denseblock = _DenseBlock(num_layers=block_length, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            num_features = num_features + block_length * growth_rate
            self.decoder_blocks.append(denseblock)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.decoder_sample = nn.ModuleList(self.decoder_sample)

        # Output
        #self.pool1 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        #self.up1 = _UpSample(num_input_features=num_features, num_output_features=num_features)
        self.norm1 = nn.BatchNorm2d(num_features)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(num_features, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        x = self.conv0(x)

        skip_connections = []
        #print x.size()
        for (denseblock, transition) in zip(self.encoder_blocks, self.encoder_sample):
            x = denseblock(x)
            skip_connections.append(x)
            x = transition(x)
        #    print x.size()
        #print "bottom"
        x = self.bottom_block(x)
        
        for (transition, denseblock, skip) in zip(self.decoder_sample, self.decoder_blocks, skip_connections[::-1]):
            x = transition(x)
            x = torch.cat([x, skip], 1)
            x = denseblock(x)
        #    print x.size()

        x = self.conv1(self.relu1(self.norm1(x)))
        #print x.size()
        return F.softmax(x)

# InceptionUNet

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionLayer(nn.Module):

    def __init__(self, input_features, hidden_features, output_features, scale=1.0):
        super(InceptionLayer, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(input_features, hidden_features, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(input_features, hidden_features, kernel_size=1, stride=1),
            BasicConv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(input_features, hidden_features, kernel_size=1, stride=1),
            BasicConv2d(hidden_features, int(hidden_features * 1.5), kernel_size=3, stride=1, padding=1),
            BasicConv2d(int(hidden_features * 1.5), hidden_features * 2, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(hidden_features * 4, output_features, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = out * self.scale + x
        out = self.conv2d(out)
        out = self.relu(out)
        return out

class InceptionBlockDown(nn.Sequential):
    def __init__(self, num_layers, input_features, output_features):
        super(InceptionBlockDown, self).__init__()
        for i in range(num_layers):
            if i==0:
                layer = InceptionLayer(input_features=input_features, hidden_features=input_features//4, output_features=output_features)
            else:
                layer = InceptionLayer(input_features=output_features, hidden_features=output_features//4, output_features=output_features)
            self.add_module('denselayer%d' % (i + 1), layer)

class InceptionBlockUp(nn.Sequential):
    def __init__(self, num_layers, input_features, output_features):
        super(InceptionBlockUp, self).__init__()
        for i in range(num_layers):
            if i==(num_layers-1):
                layer = InceptionLayer(input_features=input_features, hidden_features=input_features//4, output_features=output_features)
            else:
                layer = InceptionLayer(input_features=input_features, hidden_features=input_features//4, output_features=input_features)
            self.add_module('denselayer%d' % (i + 1), layer)

class InceptionUNet(nn.Module):

    def __init__(self, input_features=3, network_depth=5, block_length=4, num_init_features=16):
        super(InceptionUNet, self).__init__()

        # Input

        self.conv0 = nn.Conv2d(input_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        #self.down0 = transition = _DownSample(num_input_features=num_init_features, num_output_features=num_init_features)
        #self.pool0 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        num_features = num_init_features

        # Encoder

        skip_connections = []
        self.encoder_blocks = []
        self.encoder_sample = []
        for i in range(network_depth-1):
            denseblock = InceptionBlockDown(num_layers=block_length, input_features=num_features, output_features=num_features * 2)
            num_features = num_features * 2
            skip_connections.append(num_features//2)
            self.encoder_blocks.append(denseblock)
            transition = nn.MaxPool2d(2)
            self.encoder_sample.append(transition)
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.encoder_sample = nn.ModuleList(self.encoder_sample)

        # Bottom

        self.bottom_block_down = InceptionBlockDown(num_layers=block_length, input_features=num_features, output_features=num_features * 2)
        num_features = num_features * 2
        self.bottom_block_up = InceptionBlockUp(num_layers=block_length, input_features=num_features, output_features=num_features // 4)
        num_features = num_features // 2

        # Decoder

        self.decoder_blocks = []
        self.decoder_sample = []
        for i, skip in zip(range(network_depth-1), skip_connections[::-1]):
            transition = nn.UpsamplingNearest2d(scale_factor=2)
            self.decoder_sample.append(transition)
            denseblock = InceptionBlockUp(num_layers=block_length, input_features=num_features, output_features=num_features // 4)
            num_features = num_features // 2
            self.decoder_blocks.append(denseblock)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.decoder_sample = nn.ModuleList(self.decoder_sample)

        # Output
        self.conv1 = nn.Conv2d(num_features // 2, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        x = self.conv0(x)

        skip_connections = []
        #print x.size()
        for (denseblock, transition) in zip(self.encoder_blocks, self.encoder_sample):
            x = denseblock(x)
            skip_connections.append(x)
            x = transition(x)
        #    print x.size()
        #print "bottom"
        x = self.bottom_block_down(x)
        x = self.bottom_block_up(x)
        
        for (transition, denseblock, skip) in zip(self.decoder_sample, self.decoder_blocks, skip_connections[::-1]):
            x = transition(x)
            x = torch.cat([x, skip[:,::2,:,:]], 1)
            x = denseblock(x)

        x = self.conv1(x)
        return F.softmax(x)

# ResuNet

class ResLayer(nn.Module):

    def __init__(self, n_features):
        super(ResLayer, self).__init__()

        self.conv_a = BasicConv2d(n_features, n_features, kernel_size=3, stride=1, padding=1)
        self.conv_b = BasicConv2d(n_features, n_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.conv_a(x)
        y = self.conv_b(y)
        y = torch.add(y, x)
        return y

class SampleLayer(nn.Module):

    def __init__(self, n_features, down=True):
        super(SampleLayer, self).__init__()

        if down:
            n_features_out = n_features * 2
            self.pool = nn.MaxPool2d(2)
        else:
            n_features_out = n_features // 4
            self.pool = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv = BasicConv2d(n_features, n_features_out, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class ResBlock(nn.Sequential):
    def __init__(self, num_layers, n_features):
        super(ResBlock, self).__init__()
        for i in range(num_layers):
            layer = ResLayer(n_features=n_features)
            self.add_module('denselayer%d' % (i + 1), layer)

class ResuNet(nn.Module):

    def __init__(self, input_features=3, network_depth=4, block_length=2, num_init_features=16):
        super(ResuNet, self).__init__()

        # Input

        self.conv0 = BasicConv2d(input_features, num_init_features, kernel_size=3, stride=1, padding=1)
        num_features = num_init_features

        # Encoder

        skip_connections = []
        self.encoder_blocks = []
        self.encoder_sample = []
        for i in range(network_depth):
            denseblock = ResBlock(num_layers=block_length, n_features=num_features)
            self.encoder_blocks.append(denseblock)
            transition = SampleLayer(n_features=num_features, down=True)
            num_features = num_features * 2
            self.encoder_sample.append(transition)
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.encoder_sample = nn.ModuleList(self.encoder_sample)

        # Bottom

        self.bottom_block = ResBlock(num_layers=block_length, n_features=num_features)

        # Decoder

        self.decoder_blocks = []
        self.decoder_sample = []
        for i in range(network_depth):
            transition = SampleLayer(n_features=num_features, down=False)
            num_features = num_features // 2
            self.decoder_sample.append(transition)
            denseblock = ResBlock(num_layers=block_length, n_features=num_features)
            self.decoder_blocks.append(denseblock)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.decoder_sample = nn.ModuleList(self.decoder_sample)

        # Output
        self.conv1 = nn.Conv2d(num_features, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        x = self.conv0(x)

        skip_connections = []
        for (denseblock, transition) in zip(self.encoder_blocks, self.encoder_sample):
            x = denseblock(x)
            skip_connections.append(x)
            x = transition(x)


        x = self.bottom_block(x)
        
        for (transition, denseblock, skip) in zip(self.decoder_sample, self.decoder_blocks, skip_connections[::-1]):
            x = transition(x)
            x = torch.cat([x, skip[:,::2,:,:]], 1)
            x = denseblock(x)

        return F.softmax(self.conv1(x))
