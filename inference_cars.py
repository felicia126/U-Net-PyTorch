import os
import networks
import numpy as np
import torch
import time
from scipy.ndimage import zoom
from data_load import CarDataSetInference
from torch.autograd import Variable

### variables ###

# name of the model saved
model_name = 'DUNet'

# directory where to store nii.gz or numpy files
result_folder = 'results'
test_folder = 'data/test'

batch_size = 5

#################

files = os.listdir(test_folder)

# load network
cuda = torch.cuda.is_available()
net = torch.load("model_"+model_name+".pht")
if cuda: net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
net.eval() # inference mode

# data loader
cars = CarDataSetInference(image_directory=test_folder)
train_data = torch.utils.data.DataLoader(cars, batch_size=batch_size)

start = time.time()
lines = []

for i, data in enumerate(train_data):

	# wrap data in Variables
	inputs, file_names = data

	# cast to torch
	inputs = Variable(inputs, volatile=True)
	if cuda: inputs = inputs.cuda()

	# inference
	outputs = net(inputs)
	outputs = outputs.data.cpu().numpy()
	outputs = outputs[:, 1, :, :].round()

	for output, file_name in zip(outputs, file_names):
		# resize
		output = zoom(output, 4)
		output = output.round()
		# calculate run-length encoding
		pixels = output.flatten()
		pixels[0] = 0
		pixels[-1] = 0
		runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
		runs[1::2] = runs[1::2] - runs[:-1:2]
		rle = ' '.join(str(x) for x in runs)
		rle = [file_name.split('.')[0], rle ]
		lines.append(rle)
print time.time()-start

# save to csv file
import csv

with open("output.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(lines)