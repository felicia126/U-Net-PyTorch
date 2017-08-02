import os
import networks
import numpy as np
import torch
from scipy.ndimage import imread
from scipy.ndimage import zoom
from scipy.misc import imresize
from torch.autograd import Variable

### variables ###

# name of the model saved
model_name = 'DUNet'

# directory where to store nii.gz or numpy files
result_folder = 'results'
test_folder = 'data/test'

#################

files = os.listdir(test_folder)

# load network
cuda = torch.cuda.is_available()
net = torch.load("model_"+model_name+".pht")
if cuda: net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
net.eval() # inference mode

lines = []
for file_name in files:

	inputs = np.zeros((1280, 1920, 3))

	# load image and mask
	inputs[:, 0:1918, :] = imread(os.path.join(test_folder, file_name))

	# zoom in
	inputs = imresize(inputs, 0.25)

	# scale / normalize
	inputs = inputs / 255.0

	# transpose inputs and add empty axis to labels
	inputs = np.expand_dims(inputs.transpose(2,0,1), 0)

	# cast to torch
	inputs = Variable(torch.from_numpy(inputs).float(), volatile=True)
	if cuda: inputs = inputs.cuda()

	# inference
	outputs = net(inputs)
	outputs = outputs[0, 1, :, :].round()
	outputs = outputs.data.cpu().numpy()

	# resize
	outputs = zoom(outputs, 4)
	outputs = outputs.round()

	# calculate run-length encoding
	pixels = outputs.flatten()
	pixels[0] = 0
	pixels[-1] = 0
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
	runs[1::2] = runs[1::2] - runs[:-1:2]
	rle = ' '.join(str(x) for x in runs)
	rle = [file_name.split('.')[0], rle ]
	lines.append(rle)

# save to csv file
import csv

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(lines)