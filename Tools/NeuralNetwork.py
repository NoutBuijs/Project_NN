import numpy as np
from tqdm import tqdm
import copy

"""
# Project_NN
Creation of a neural network form scratch by me as an introduction to the world of AI.

As a heads up for the reader I started this project not with the intent of creating a state 
of the art neural network; the classes and functions in the tools folder form by no means 
a perfectly designed AI. Lastly, the fundamentals of the network, have been retrieved from 
the content of 3Blue1Brown's lessons on neural networks.
"""


def leak_relu(leak_factor,x):                                   # leaky relu
    values = copy.deepcopy(x)
    values[np.where(values < 0)] = leak_factor * values[np.where(values < 0 )]
    return values

def d_leak_relu(leak_factor,x):                                 # derivative of leaky relu
    values = copy.deepcopy(x)
    values[np.where(values >= 0)] = 1
    values[np.where(values < 0)] = leak_factor
    return values

def relu(x):                                                    # relu
    values = copy.deepcopy(x)
    values[np.where(values < 0)] = 0
    return values

def d_relu(x):                                                  # derivative of relu
    values = copy.deepcopy(x)
    values[np.where(values >= 0)] = 1
    values[np.where(values < 0)] = 0
    return values

def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))                               # sigmoid

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))                        # derivative of the sigmoid

def forward_flow(weights, biases, neurons,                      # forward process of data
                 activation_function = sigmoid):

    """ weights = matrix with #neurons number of columns and #neurons_next_layer number of rows
        biases = array of #neurons_next_layer number of entries
        neurons = transpose array of scaled neurons; vector form """

    return activation_function(np.array(np.dot(weights, neurons) + biases,dtype=float))

def cost(answer, solution):                                     # cost function

    """ answer = array with answer form the neural network;     N amount of entries
        solution = array with solution to the problem;          N amount of entries """

    return np.sum((answer-solution) ** 2)

class zeros:

    # initialize object
    def __init__(self,                                          # input variables
                 size_of_inputlayer,
                 size_of_outputlayer,
                 neurons_per_hidden_layer = [16,16],
                 activation_fuction = sigmoid,
                 derivative_activation_function = d_sigmoid):

        # renaming variables
        self.SOI    =   size_of_inputlayer
        self.NOHL   =   np.shape(neurons_per_hidden_layer)[0]
        self.NPHL   =   neurons_per_hidden_layer
        self.SOO    =   size_of_outputlayer
        self.AF     =   activation_fuction
        self.DAF    =   derivative_activation_function

        # computing variables
        self.NOW    =   self.SOI * self.NPHL[0]                 # number of weights;    between n0 and n1
        self.LS     =   np.append(self.SOI, self.NPHL)          # layer sizes;          n0 and n_hidden
        self.LS     =   np.append(self.LS, self.SOO)            # layer sizes;          all layers

        # creating array containing number of weights per layer
        self.NWPL    =  np.array([0])                           # number of weights per layer
        for i in range(0,np.size(self.LS)-1):
            self.NWPL = np.append(self.NWPL, self.LS[i] * self.LS[i+1])

        # creating array containing number of biases per layer
        self.NBPL   = np.array([0])                             # number of biases per layer
        for i in range(0,np.size(self.LS)-1):
            self.NBPL = np.append(self.NBPL,self.LS[i+1])


        # update self.NOW
        for n in range(1,np.size(self.LS)-1):                   # number of weights; total
            self.NOW += self.LS[n] * self.LS[n+1]

        # create weights, biases and neuron arrays
        self.W      =   np.zeros(self.NOW)                      # all weights of the network stored in one array
        self.B      =   np.zeros(np.sum(self.LS[1:]))           # all biases of the network stored in one array
        self.N      =   np.zeros(np.sum(self.LS))               # all neurons of the network stored in one array

    def predict_tqdm(self, input_arrays):                                   # prediction of the NN based on input; prediction is the answer of the NN

        """ input = array of input values for the NN with #input neurons as size
            return = answer to the input"""


        compatibility = True
        for input in input_arrays:
            if np.size(input) != self.LS[0]:
                compatibility = False

        if compatibility:
            output_array = np.zeros(np.shape(input_arrays)[0],dtype=object)
            n = 0
            for input in tqdm(range(0,np.shape(input_arrays)[0])):
                n_i = input_arrays[input]

                for i in range(1,np.size(self.LS)):

                    # retrieve weights corresponding to layer form weights array
                    w_i = np.split(self.W[np.sum(self.NWPL[:i]): \
                                              np.sum(self.NWPL[:i + 1])], self.LS[i])

                    # retrieve biases corresponding to layer form biases array
                    b_i = self.B[np.sum(self.NBPL[:i]): \
                                 np.sum(self.NBPL[:i + 1])]
                    #return w_i, b_i , n_i
                    # computing new neuron layer values
                    n_i = forward_flow(w_i,b_i,n_i,
                                       activation_function=self.AF)

                    # store output arrays
                    output_array[n] = n_i
                n += 1
            return output_array

        else:

            print("not compatible input array: \n",
                  f"expected size: {self.LS[0]} \n",
                  f"input size: {np.size(input)}")

    def predict(self, input_arrays):                                   # prediction of the NN based on input; prediction is the answer of the NN

        """ input = array of input values for the NN with #input neurons as size
            return = answer to the input"""

        compatibility = True
        for input in input_arrays:
            if np.size(input) != self.LS[0]:
                compatibility = False

        if compatibility:
            output_array = np.zeros(np.shape(input_arrays)[0],dtype=object)
            n = 0
            for input in range(0,np.shape(input_arrays)[0]):
                n_i = input_arrays[input]

                for i in range(1,np.size(self.LS)):

                    # retrieve weights corresponding to layer form weights array
                    w_i = np.split(self.W[np.sum(self.NWPL[:i]): \
                                              np.sum(self.NWPL[:i + 1])], self.LS[i])

                    # retrieve biases corresponding to layer form biases array
                    b_i = self.B[np.sum(self.NBPL[:i]): \
                                 np.sum(self.NBPL[:i + 1])]
                    #return w_i, b_i , n_i
                    # computing new neuron layer values
                    n_i = forward_flow(w_i,b_i,n_i,
                                       activation_function=self.AF)

                    # store output arrays
                    output_array[n] = n_i
                n += 1
            return output_array

        else:

            print("not compatible input array: \n",
                  f"expected size: {self.LS[0]} \n",
                  f"input size: {np.size(input)}")

    def solve(self, input):                                   # prediction of the NN based on input; designed for learn method

        """ input = array of input values for the NN with #input neurons as size
            return = answer to the input, all neuron values"""

        if np.size(input) == self.LS[0]:
            n_array = np.zeros(np.size(self.LS), dtype=object)
            n_i = input

            for i in range(1,np.size(self.LS)):
                # stores neuron values in array of arrays
                n_array[i - 1] = n_i

                # retrieve weights corresponding to layer form weights array
                w_i = np.split(self.W[np.sum(self.NWPL[:i]): \
                                          np.sum(self.NWPL[:i + 1])], self.LS[i])

                # retrieve biases corresponding to layer form biases array
                b_i = self.B[np.sum(self.NBPL[:i]): \
                             np.sum(self.NBPL[:i + 1])]

                # computing new neuron layer values
                n_i = forward_flow(w_i,b_i,n_i,
                                   activation_function=self.AF)

            # stores last neuron values in array
            n_array[-1] = n_i
            return n_i, n_array

    def learn_tqdm(self,input_arrays,solution_arrays):

        # compatibility check for input and output arrays w.r.t. NN
        compatibility = True
        for i in range(0, np.shape(input_arrays)[0]):

            if np.size(input_arrays[i]) != self.LS[0] or np.size(solution_arrays[i]) != self.LS[-1]:
                idx = i
                compatibility = False

        # input, output array consistency; same size
        consistency = True
        if np.shape(input_arrays)[0] != np.shape(solution_arrays)[0]:
            consistency = False

        if consistency and compatibility:

            # arrays that will be storing requested increments form all test samples; averaged for final increments
            dW = np.zeros(np.size(solution_arrays), dtype=object)
            dB = np.zeros(np.size(solution_arrays), dtype=object)
            C  = np.zeros(np.size(solution_arrays), dtype=float)

            for i in tqdm(range(np.shape(input_arrays)[0])):

                # retrieve prediction, all neuron values
                a, n_arrays = self.solve(input_arrays[i])

                # cost of prediction
                C[i] = cost(a,solution_arrays[i])

                # initial weight indices
                w_idx = np.array([np.size(self.W) - self.NWPL[-1],
                                  np.size(self.W)])

                # initial bias indices
                b_idx = np.array([np.size(self.B) - self.LS[-1],
                                  np.size(self.B)])

                # weight, biases increments arrays
                dw = np.zeros(np.size(self.LS) - 1, dtype=object)
                db = np.zeros(np.size(self.LS) - 1, dtype=object)

                # retrieving previous neuron values, weights and biases to initialize delta
                a_p = np.reshape(n_arrays[-2],(np.size(n_arrays[-2]),1))
                w = np.reshape(self.W[w_idx[0]: w_idx[1]],(np.size(a),np.size(a_p)))
                b = np.reshape(self.B[b_idx[0]: b_idx[1]],(np.size(self.B[b_idx[0]: b_idx[1]]),1))
                delta = self.DAF(np.dot(w,a_p) + b) * np.reshape(2 * (a - solution_arrays[i]),
                                                                                       (np.size(a),1))

                # back propagation; working though layers backwards
                # run up to # of layers - 1 to prevent range issues with a_pp
                for l in range(1, np.size(self.LS)-1):

                    # update weights indices
                    w_idx[0] += - self.NWPL[-l - 1]
                    w_idx[1] += - self.NWPL[-l]

                    # update biases indices
                    b_idx[0] += -self.LS[-l - 1]
                    b_idx[1] += -self.LS[-l]

                    # retrieve neurons corresponding to layer from neuron array
                    a_pp = np.reshape(n_arrays[-l - 2],(np.size(n_arrays[-l - 2]),1))

                    # retrieve weights of previous layer
                    wp = np.reshape(self.W[w_idx[0]: w_idx[1]],(np.size(a_p),np.size(a_pp)))

                    # retrieve biases of previous layer
                    bp = np.reshape(self.B[b_idx[0]: b_idx[1]],(np.size(self.B[b_idx[0]: b_idx[1]]),1))

                    dw[-l] = np.concatenate(delta * a_p.T)
                    db[-l] = np.concatenate(delta)
                    delta = np.dot(np.reshape(self.DAF(np.dot(wp,a_pp) + bp),
                                              (np.size(a_p),1)) * w.T, delta)

                    a_p = a_pp
                    w = wp

                # solve boundary
                dw[0] = np.concatenate(delta * a_p.T)
                db[0] = np.concatenate(delta)

                dW[i] = np.concatenate(dw)
                dB[i] = np.concatenate(db)

            self.W -= np.mean(dW, axis=0)
            self.B -= np.mean(dB, axis=0)

        else:
            if not compatibility:
                print("not compatible arrays: \n",
                      f"expected input size: {self.LS[0]} \n",
                      f"given input size: {np.size(input_arrays[idx])} \n",
                      f"expected solution size: {self.LS[-1]} \n",
                      f"given solution array size: {np.size(solution_arrays[idx])} \n")
            if not consistency:
                print("not consistent number of arrays given: \n",
                      f"given number of input arrays:  {np.shape(input_arrays)[0]} \n",
                      f"given number of solution arrays: {np.shape(solution_arrays)[0]}")

        return np.max(C), np.mean(C), np.min(C)

    def learn(self, input_arrays, solution_arrays):

        # compatibility check for input and output arrays w.r.t. NN
        compatibility = True
        for i in range(0, np.shape(input_arrays)[0]):

            if np.size(input_arrays[i]) != self.LS[0] or np.size(solution_arrays[i]) != self.LS[-1]:
                idx = i
                compatibility = False

        # input, output array consistency; same size
        consistency = True
        if np.shape(input_arrays)[0] != np.shape(solution_arrays)[0]:
            consistency = False

        if consistency and compatibility:

            # arrays that will be storing requested increments form all test samples; averaged for final increments
            dW = np.zeros(np.size(solution_arrays), dtype=object)
            dB = np.zeros(np.size(solution_arrays), dtype=object)
            C  = np.zeros(np.size(solution_arrays), dtype=float)

            for i in range(np.shape(input_arrays)[0]):

                # retrieve prediction, all neuron values
                a, n_arrays = self.solve(input_arrays[i])

                # cost of prediction
                C[i] = cost(a,solution_arrays[i])

                # initial weight indices
                w_idx = np.array([np.size(self.W) - self.NWPL[-1],
                                  np.size(self.W)])

                # initial bias indices
                b_idx = np.array([np.size(self.B) - self.LS[-1],
                                  np.size(self.B)])

                # weight, biases increments arrays
                dw = np.zeros(np.size(self.LS) - 1, dtype=object)
                db = np.zeros(np.size(self.LS) - 1, dtype=object)

                # retrieving previous neuron values, weights and biases to initialize delta
                a_p = np.reshape(n_arrays[-2],(np.size(n_arrays[-2]),1))
                w = np.reshape(self.W[w_idx[0]: w_idx[1]],(np.size(a),np.size(a_p)))
                b = np.reshape(self.B[b_idx[0]: b_idx[1]],(np.size(self.B[b_idx[0]: b_idx[1]]),1))
                delta = self.DAF(np.dot(w,a_p) + b) * np.reshape(2 * (a - solution_arrays[i]),
                                                                                       (np.size(a),1))

                # back propagation; working though layers backwards
                # run up to # of layers - 1 to prevent range issues with a_pp
                for l in range(1, np.size(self.LS)-1):

                    # update weights indices
                    w_idx[0] += - self.NWPL[-l - 1]
                    w_idx[1] += - self.NWPL[-l]

                    # update biases indices
                    b_idx[0] += -self.LS[-l - 1]
                    b_idx[1] += -self.LS[-l]

                    # retrieve neurons corresponding to layer from neuron array
                    a_pp = np.reshape(n_arrays[-l - 2],(np.size(n_arrays[-l - 2]),1))

                    # retrieve weights of previous layer
                    wp = np.reshape(self.W[w_idx[0]: w_idx[1]],(np.size(a_p),np.size(a_pp)))

                    # retrieve biases of previous layer
                    bp = np.reshape(self.B[b_idx[0]: b_idx[1]],(np.size(self.B[b_idx[0]: b_idx[1]]),1))

                    dw[-l] = np.concatenate(delta * a_p.T)
                    db[-l] = np.concatenate(delta)
                    delta = np.dot(np.reshape(self.DAF(np.dot(wp,a_pp) + bp),
                                              (np.size(a_p),1)) * w.T, delta)

                    # replacing old variables for next iteration
                    a_p = a_pp
                    w = wp

                # solve boundary
                dw[0] = np.concatenate(delta * a_p.T)
                db[0] = np.concatenate(delta)


                dW[i] = np.concatenate(dw)
                dB[i] = np.concatenate(db)

            self.W -= np.mean(dW, axis=0)
            self.B -= np.mean(dB, axis=0)

        else:
            if not compatibility:
                print("not compatible arrays: \n",
                      f"expected input size: {self.LS[0]} \n",
                      f"given input size: {np.size(input_arrays[idx])} \n",
                      f"expected solution size: {self.LS[-1]} \n",
                      f"given solution array size: {np.size(solution_arrays[idx])} \n")
            if not consistency:
                print("not consistent number of arrays given: \n",
                      f"given number of input arrays:  {np.shape(input_arrays)[0]} \n",
                      f"given number of solution arrays: {np.shape(solution_arrays)[0]}")

        return np.max(C), np.mean(C), np.min(C)


class random(zeros):

    # initialize object
    def __init__(self,                                          # input variables
                 size_of_inputlayer,
                 size_of_outputlayer,
                 neurons_per_hidden_layer = [16,16],
                 activation_fuction = sigmoid,
                 derivative_activation_function = d_sigmoid):

        # renaming variables
        self.SOI    =   size_of_inputlayer
        self.NOHL   =   np.shape(neurons_per_hidden_layer)[0]
        self.NPHL   =   neurons_per_hidden_layer
        self.SOO    =   size_of_outputlayer
        self.AF     =   activation_fuction
        self.DAF    =   derivative_activation_function

        # computing variables
        self.NOW    =   self.SOI * self.NPHL[0]                 # number of weights;    between n0 and n1
        self.LS     =   np.append(self.SOI, self.NPHL)          # layer sizes;          n0 and n_hidden
        self.LS     =   np.append(self.LS, self.SOO)            # layer sizes;          all layers

        # creating array containing number of weights per layer
        self.NWPL    =  np.array([0])                           # number of weights per layer
        for i in range(0,np.size(self.LS)-1):
            self.NWPL = np.append(self.NWPL, self.LS[i] * self.LS[i+1])

        # creating array containing number of biases per layer
        self.NBPL   = np.array([0])                             # number of biases per layer
        for i in range(0,np.size(self.LS)-1):
            self.NBPL = np.append(self.NBPL,self.LS[i+1])


        # update self.NOW
        for n in range(1,np.size(self.LS)-1):                   # number of weights; total
            self.NOW += self.LS[n] * self.LS[n+1]

        # create weights, biases and neuron arrays
        self.W      =   np.random.uniform(low=-0.5,
                                          high=0.5,
                                          size=self.NOW)        # all weights of the network stored in one array
        self.B      =   np.random.uniform(low=-0.5,
                                          high=0.5,
                                          size=np.sum(self.LS[1:]))        # all biases of the network stored in one array
        self.N      =   np.zeros(np.sum(self.LS))               # all neurons of the network stored in one array

class he(zeros):

    # initialize object
    def __init__(self,                                          # input variables
                 size_of_inputlayer,
                 size_of_outputlayer,
                 neurons_per_hidden_layer = [16,16],
                 he_bias = False,
                 activation_fuction = sigmoid,
                 derivative_activation_function = d_sigmoid):

        # renaming variables
        self.SOI    =   size_of_inputlayer
        self.NOHL   =   np.shape(neurons_per_hidden_layer)[0]
        self.NPHL   =   neurons_per_hidden_layer
        self.SOO    =   size_of_outputlayer
        self.AF     =   activation_fuction
        self.DAF    =   derivative_activation_function

        # computing variables
        self.NOW    =   self.SOI * self.NPHL[0]                 # number of weights;    between n0 and n1
        self.LS     =   np.append(self.SOI, self.NPHL)          # layer sizes;          n0 and n_hidden
        self.LS     =   np.append(self.LS, self.SOO)            # layer sizes;          all layers

        # creating array containing number of weights per layer
        self.NWPL    =  np.array([0])                           # number of weights per layer
        for i in range(0,np.size(self.LS)-1):
            self.NWPL = np.append(self.NWPL, self.LS[i] * self.LS[i+1])

        # creating array containing number of biases per layer
        self.NBPL   = np.array([0])                             # number of biases per layer
        for i in range(0,np.size(self.LS)-1):
            self.NBPL = np.append(self.NBPL,self.LS[i+1])


        # update self.NOW
        for n in range(1,np.size(self.LS)-1):                   # number of weights; total
            self.NOW += self.LS[n] * self.LS[n+1]

        #initialize weights and biases array
        self.W = np.zeros(self.NOW)
        self.B = np.zeros(np.sum(self.LS[1:]))


        # create weights, biases and neuron arrays
        if he_bias:
            for l in range(1,np.size(self.LS)):

                # all weights of the network stored in one array
                self.W[np.sum(self.NWPL[:l]):np.sum(self.NWPL[:l + 1])]     =   \
                    np.random.normal(loc=0,
                                     scale=np.sqrt(2 / self.LS[l- 1]),
                                     size=self.NWPL[l])

                # all biases of the network stored in one array
                self.B[np.sum(self.NBPL[:l]):np.sum(self.NBPL[:l + 1])]      =   \
                    np.random.normal(loc=0,
                                     scale=np.sqrt(2 / self.LS[l - 1]),
                                     size=self.LS[l])
        else:
            for l in range(1,np.size(self.LS)):
                # all weights of the network stored in one array
                self.W[np.sum(self.NWPL[:l]):np.sum(self.NWPL[:l + 1])]     =   \
                    np.random.normal(loc=0,
                                     scale=np.sqrt(2 / self.LS[l- 1]),
                                     size=self.NWPL[l])


        self.N      =   np.zeros(np.sum(self.LS))               # all neurons of the network stored in one array

class max_curv(zeros):

    # initialize object
    def __init__(self,                                          # input variables
                 size_of_inputlayer,
                 size_of_outputlayer,
                 neurons_per_hidden_layer = [16,16],
                 max_curv_bias = False,
                 a = 1.36375,
                 activation_fuction = sigmoid,
                 derivative_activation_function = d_sigmoid):

        # renaming variables
        self.SOI    =   size_of_inputlayer
        self.NOHL   =   np.shape(neurons_per_hidden_layer)[0]
        self.NPHL   =   neurons_per_hidden_layer
        self.SOO    =   size_of_outputlayer
        self.a      =   a
        self.AF     =   activation_fuction
        self.DAF    =   derivative_activation_function

        # computing variables
        self.NOW    =   self.SOI * self.NPHL[0]                 # number of weights;    between n0 and n1
        self.LS     =   np.append(self.SOI, self.NPHL)          # layer sizes;          n0 and n_hidden
        self.LS     =   np.append(self.LS, self.SOO)            # layer sizes;          all layers

        # creating array containing number of weights per layer
        self.NWPL    =  np.array([0])                           # number of weights per layer
        for i in range(0,np.size(self.LS)-1):
            self.NWPL = np.append(self.NWPL, self.LS[i] * self.LS[i+1])

        # creating array containing number of biases per layer
        self.NBPL   = np.array([0])                             # number of biases per layer
        for i in range(0,np.size(self.LS)-1):
            self.NBPL = np.append(self.NBPL,self.LS[i+1])


        # update self.NOW
        for n in range(1,np.size(self.LS)-1):                   # number of weights; total
            self.NOW += self.LS[n] * self.LS[n+1]

        #initialize weights and biases array
        self.W = np.zeros(self.NOW)
        self.B = np.zeros(np.sum(self.LS[1:]))


        # create weights, biases and neuron arrays
        if max_curv_bias:
            for l in range(1,np.size(self.LS)):

                # all weights of the network stored in one array
                self.W[np.sum(self.NWPL[:l]):np.sum(self.NWPL[:l + 1])]     =   \
                    np.random.normal(loc=0,
                                     scale=np.sqrt(self.a / self.LS[l- 1]),
                                     size=self.NWPL[l])

                # all biases of the network stored in one array
                self.B[np.sum(self.NBPL[:l]):np.sum(self.NBPL[:l + 1])]      =   \
                    np.random.normal(loc=0,
                                     scale=np.sqrt(self.a / self.LS[l - 1]),
                                     size=self.LS[l])
        else:
            for l in range(1,np.size(self.LS)):
                # all weights of the network stored in one array
                self.W[np.sum(self.NWPL[:l]):np.sum(self.NWPL[:l + 1])]     =   \
                    np.random.normal(loc=0,
                                     scale=np.sqrt(self.a / self.LS[l- 1]),
                                     size=self.NWPL[l])


        self.N      =   np.zeros(np.sum(self.LS))               # all neurons of the network stored in one array

class normal(zeros):

    # initialize object
    def __init__(self,                                          # input variables
                 size_of_inputlayer,
                 size_of_outputlayer,
                 neurons_per_hidden_layer = [16,16],
                 activation_fuction = sigmoid,
                 derivative_activation_function = d_sigmoid):

        # renaming variables
        self.SOI    =   size_of_inputlayer
        self.NOHL   =   np.shape(neurons_per_hidden_layer)[0]
        self.NPHL   =   neurons_per_hidden_layer
        self.SOO    =   size_of_outputlayer
        self.AF     =   activation_fuction
        self.DAF    =   derivative_activation_function

        # computing variables
        self.NOW    =   self.SOI * self.NPHL[0]                 # number of weights;    between n0 and n1
        self.LS     =   np.append(self.SOI, self.NPHL)          # layer sizes;          n0 and n_hidden
        self.LS     =   np.append(self.LS, self.SOO)            # layer sizes;          all layers

        # creating array containing number of weights per layer
        self.NWPL    =  np.array([0])                           # number of weights per layer
        for i in range(0,np.size(self.LS)-1):
            self.NWPL = np.append(self.NWPL, self.LS[i] * self.LS[i+1])

        # creating array containing number of biases per layer
        self.NBPL   = np.array([0])                             # number of biases per layer
        for i in range(0,np.size(self.LS)-1):
            self.NBPL = np.append(self.NBPL,self.LS[i+1])


        # update self.NOW
        for n in range(1,np.size(self.LS)-1):                   # number of weights; total
            self.NOW += self.LS[n] * self.LS[n+1]

        # create weights, biases and neuron arrays

        # all weights of the network stored in one array
        self.W      =   np.random.normal(size=self.NOW)

        # all biases of the network stored in one array
        self.B      =   np.random.normal(size=np.sum(self.LS[1:]))

        # all neurons of the network stored in one array
        self.N      =   np.zeros(np.sum(self.LS))