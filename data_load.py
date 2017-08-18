import os
import torch
import numpy as np
import torch.utils.data as data_utils
from scipy.ndimage import imread
from scipy.misc import imresize

# Liver Dataset - segmentation task
# when false selects both the liver and the tumor as positive labels
class CarDataSet(torch.utils.data.Dataset):

    def __init__(self, image_directory, mask_directory, zoom=0.5, context=False):

        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.image_files = os.listdir(image_directory)
        self.mask_files = os.listdir(mask_directory)
        self.image_files.sort()
        self.mask_files.sort()
        self.zoom = zoom
        self.context = context

    def __getitem__(self, idx):

        # zoom factor
        z = self.zoom

        # random vertical offset 0, 1 or 2
        r = int(np.random.rand()*3)

        if self.context:
            # get label from index
            labels = self.__getlabel__(idx, r)
            labels = torch.from_numpy(labels).long()
            # get inputs from images turned slightly to the left and right
            if idx%16 == 0:
                idx = [idx+15, idx, idx+1]
            elif (idx-15)%16 == 0:
                idx = [idx-1, idx, idx-15]
            else:
                idx = [idx-1, idx, idx+1]
            # get all images and concat them
            inputs = []
            for i in idx:
                inputs.append(self.__getinput__(i, r))
            inputs = np.concatenate(inputs, axis=0)
            inputs = torch.from_numpy(inputs).float()
            return (inputs, labels)
        else:
            # just return the regular getitem
            inputs = self.__getinput__(idx, r)
            labels = self.__getlabel__(idx, r)
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).long()
            return (inputs, labels)

    def __getinput__(self, idx, r):

        inputs = np.zeros((1280, 1920, 3))
        inputs[:, 0+r:1918+r, :] = imread(os.path.join(self.image_directory, self.image_files[idx]))
        inputs = imresize(inputs, 0.1)
        inputs = inputs / 255.0
        inputs = inputs.transpose(2,0,1)
        return inputs

    def __getlabel__(self, idx, r):

        labels = np.zeros((1280, 1920))
        labels[:, 0+r:1918+r] = imread(os.path.join(self.mask_directory, self.mask_files[idx]))[:, :, 0]
        labels = imresize(labels, 0.1)
        labels = labels / 255
        labels = np.expand_dims(labels, 0)
        return labels

    def __len__(self):

        return len(self.image_files)

class CarDataSetInference(torch.utils.data.Dataset):

    def __init__(self, image_directory):

        self.image_directory = image_directory
        self.image_files = os.listdir(image_directory)

    def __getitem__(self, idx):

        file_name = self.image_files[idx]
        inputs = np.zeros((1280, 1920, 3))
        inputs[:, 0:1918, :] = imread(os.path.join(self.image_directory, file_name))
        inputs = imresize(inputs, 0.25)
        inputs = inputs / 255.0
        inputs = inputs.transpose(2,0,1)
        inputs = torch.from_numpy(inputs).float()
        return (inputs, file_name)

    def __len__(self):

        return len(self.image_files)

# Liver Dataset - segmentation task
# when false selects both the liver and the tumor as positive labels
class LiverDataSet(torch.utils.data.Dataset):

    def __init__(self, directory, augment=False, context=0):

        self.augment = augment
        self.context = context
        self.directory = directory
        self.data_files = os.listdir(directory)

        def get_type(s): return s[:1]
        def get_item(s): return int(s.split("_")[1].split(".")[0])
        def get_patient(s): return int(s.split("-")[1].split("_")[0])

        self.data_files.sort(key = lambda x: (get_type(x), get_patient(x), get_item(x)))
        self.data_files = zip(self.data_files[len(self.data_files)/2:], self.data_files[:len(self.data_files)/2])
    
    def __getitem__(self, idx):

        if self.context > 0:
            return load_file_context(self.data_files, idx, self.context, self.directory, self.augment)
        else:
            return load_file(self.data_files[idx], self.directory, self.augment)

    def __len__(self):

        return len(self.data_files)

    def getWeights(self):

        weights = []
        pos = 0.0
        neg = 0.0

        for data_file in self.data_files:

            _, labels = data_file
            labels = np.load(os.path.join(self.directory, labels))

            if labels.sum() > 0:
                weights.append(-1)
                pos += 1
            else:
                weights.append(0)
                neg += 1

        weights = np.array(weights).astype(float)
        weights[weights==0] = 1.0 / neg * 0.1
        weights[weights==-1] = 1.0 / pos * 0.9

        print('%d samples with positive labels, %d samples with negative labels.' % (pos, neg))

        return weights

    def getPatients(self):

        patient_dictionary = {}

        for i, data_file in enumerate(self.data_files):

            _, labels = data_file
            patient = labels.split("_")[0].split("-")[1]

            if patient in patient_dictionary:
                patient_dictionary[patient].append(i)
            else:
                patient_dictionary[patient] = [i]

        return patient_dictionary


# load data_file in directory and possibly augment
def load_file(data_file, directory, augment):

    inputs, labels = data_file
    inputs, labels = np.load(os.path.join(directory, inputs)), np.load(os.path.join(directory, labels))
    inputs, labels = np.expand_dims(inputs, 0), np.expand_dims(labels, 0)

    # augment
    if augment and np.random.rand() > 0.5:
        inputs = np.fliplr(inputs).copy()
        labels = np.fliplr(labels).copy()

    features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
    return (features, targets)

# load data_file in directory and possibly augment including the slides above and below it
def load_file_context(data_files, idx, context, directory, augment):

    # check whether all inputs need to be augmented
    if augment and np.random.rand() > 0.5: augment = False

    # load middle slice
    inputs_b, labels_b = data_files[idx]
    inputs_b, labels_b = np.load(os.path.join(directory, inputs_b)), np.load(os.path.join(directory, labels_b))
    inputs_b, labels_b = np.expand_dims(inputs_b, 0), np.expand_dims(labels_b, 0)

    # augment
    if augment:
        inputs_b = np.fliplr(inputs_b).copy()
        labels_b = np.fliplr(labels_b).copy()

    # load slices before middle slice
    inputs_a = []
    for i in range(idx-context, idx):

        # if different patient or out of bounds, take middle slice, else load slide
        if i < 0 or data_files[idx][0][:-6] != data_files[i][0][:-6]:
            inputs = inputs_b
        else:
            inputs, _ = data_files[i]
            inputs = np.load(os.path.join(directory, inputs))
            inputs = np.expand_dims(inputs, 0)
            if augment: inputs = np.fliplr(inputs).copy()

        inputs_a.append(inputs)

    # load slices after middle slice
    inputs_c = []
    for i in range(idx+1, idx+context+1):

        # if different patient or out of bounds, take middle slice, else load slide
        if i >= len(data_files) or data_files[idx][0][:-6] != data_files[i][0][:-6]:
            inputs = inputs_b
        else:
            inputs, _ = data_files[i]
            inputs = np.load(os.path.join(directory, inputs))
            inputs = np.expand_dims(inputs, 0)
            if augment: inputs = np.fliplr(inputs).copy()

        inputs_c.append(inputs)

    # concatenate all slices for context
    # middle sice first, because the network that one for the residual connection
    inputs = [inputs_b] + inputs_a + inputs_c
    labels = labels_b

    inputs = np.concatenate(inputs, 0)

    features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
    return (features, targets)
