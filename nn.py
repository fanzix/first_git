import random
import cPickle
import gzip
import numpy as np
import json

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        wb=zip(self.biases, self.weights)
        b1,w1 = wb[0]
        a = sigmoid(np.dot(w1, a)+b1)
        b2,w2 = wb[1]
        a = softmax(np.dot(w2, a)+b2)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] 
        zs = [] 

        wb=zip(self.biases, self.weights)

        b1,w1 = wb[0]
        z = np.dot(w1, activation)+b1
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
        
        b2,w2 = wb[1]
        z2 = np.dot(w2, activation)+b2
        zs.append(z2)
        activation = softmax(z2)
        activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) 

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        sp = sigmoid_prime(zs[0])
        delta = np.dot(self.weights[1].transpose(), delta) * sp
        nabla_b[-2] = delta
        nabla_w[-2] = np.dot(delta, activations[0].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations-y

    def save_parameter(self,filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
               }
        with open(filename, "wb") as f:
            json.dump(data, f)
        f.close()


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
