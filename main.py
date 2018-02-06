import load_data
import nn
import logistic_net

import random
import cv2
import numpy as np
import json
import os

command = "python tf_logistic.py"
command2 = "python tf_nn.py"
command3 = "python cnn.py"
command4 = "train.py"

os.system(command)
os.system(command2)
os.system(command3)
os.system(command4)

USPS_data = load_data.USPS_data_loader()

l_filename = 'parameter/logistic_epoch_60_batch_100.json'
nn_filename = 'parameter/nn_epoch_60_batch_10_neutron_30.json'

def load_parameter(file):
	f = open(file, "r")
	data = json.load(f)
	f.close()

	weights = [np.array(w) for w in data["weights"]]
	biases = [np.array(b) for b in data["biases"]]
	wb = zip(biases,weights)
	return wb

def logistic_feedforward(a): 
	for b,w in load_parameter(l_filename):
		a = logistic_net.softmax(np.dot(w, a)+b)
	return a

def nn_feedforward(a):
	wb=load_parameter(nn_filename)
	b1,w1 = wb[0]
	a = nn.sigmoid(np.dot(w1, a)+b1)
	b2,w2 = wb[1]
	a = nn.softmax(np.dot(w2, a)+b2)
	return a

def evaluate(test_data):
	test_results1 = [(np.argmax(logistic_feedforward(x)), np.argmax(y))
				for (x, y) in test_data]

	test_results2 = [(np.argmax(nn_feedforward(x)), np.argmax(y))
				for (x, y) in test_data]

	result1=sum(int(x == y) for (x, y) in test_results1)
	result2=sum(int(x == y) for (x, y) in test_results2)
	result=(result1,result2)
	return result

random.shuffle(USPS_data)
result = evaluate(USPS_data)

print("code of our own")
print("Test USPS dataset")
print("The accuracy of logictic regression is: "+str(result[0])/20000.0)
print("The accuracy of neutral network is: "+str(result[1])/20000.0)
