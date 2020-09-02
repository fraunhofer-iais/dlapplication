import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")

from dlutils.models.pytorch.MNISTNetwork import MnistNet
from DLplatform.parameters.pyTorchNNParameters import PyTorchNNParameters
import pickle
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
	
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

testset = torchvision.datasets.MNIST(root='mnist_pytorch_data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=2)

param = pickle.loads(open("coordinator/currentAveragedState", "rb").read())
net = MnistNet().cuda()
state_dict = OrderedDict()
for k,v in param.get().items():
	if v.shape == ():
		state_dict[k] = torch.tensor(v)
	else:
		state_dict[k] = torch.cuda.FloatTensor(v)
net.load_state_dict(state_dict)

net.eval()
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data[0].cuda(), data[1].cuda()
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * correct / total))



