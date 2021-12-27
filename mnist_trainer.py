import torchvision
from Tools import NeuralNetwork as NN
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
np.set_printoptions(precision=3)

mnist_train = torchvision.datasets.MNIST(r"mnist_data",train=True)
mnist_train_data = mnist_train.train_data.numpy()
mnist_train_labels = mnist_train.test_labels.numpy()

shape = (np.shape(mnist_train_data)[0],np.shape(mnist_train_data)[1]**2)
mnist_train_data = np.reshape(mnist_train_data,shape)
mnist_train_data = normalize(mnist_train_data,norm="max")

labels = np.zeros(np.shape(mnist_train_labels)[0],dtype=object)
for i,label in enumerate(mnist_train_labels):
    label_empty = np.zeros(10)
    label_empty[label] = 1
    labels[i] = label_empty
labels = np.vstack(labels)

nn = NN.he(np.shape(mnist_train_data)[1],np.shape(labels)[1],
           activation_fuction=lambda x:NN.leak_relu(0.01, x),
           derivative_activation_function=lambda x:NN.d_leak_relu(0.01, x))

epochs=300
sets = 4
batchsize = int(np.round(np.shape(labels)[0]/epochs))
cmax = np.zeros(epochs*sets)
cavg = np.zeros(epochs*sets)
cmin = np.zeros(epochs*sets)
epoch = np.zeros(epochs*sets)

counter = 0
for j in range(int(sets)):
    print(f"set: {j+1} out of {sets}")
    n = 0
    for i in tqdm(range(int(np.ceil(np.shape(labels)[0])))):

        if i//batchsize == n:
            input = mnist_train_data[i:i+batchsize]
            solution = labels[i:i+batchsize]
            c1, c2, c3 = nn.learn(input,solution)
            cmax[counter] = c1
            cavg[counter] = c2
            cmin[counter] = c3
            epoch[counter] = counter
            n += 1
            counter += 1

plt.title("cost over epoch")
plt.plot(epoch, cmax,c="red")
plt.plot(epoch, cavg, c="black")
plt.plot(epoch, cmin, c="green")

mnist_test = torchvision.datasets.MNIST(r"mnist_data",train=False)
mnist_test_data = mnist_test.test_data.numpy()
mnist_test_labels = mnist_test.test_labels.numpy()
shape_test = (np.shape(mnist_test_data)[0],np.shape(mnist_test_data)[1]**2)
mnist_test_data = np.reshape(mnist_test_data,shape_test)
mnist_test_data = normalize(mnist_test_data,norm="max")

test = nn.predict_tqdm(mnist_test_data)

output = np.zeros(np.shape(test)[0])
mistakes = []
for i in range(np.shape(test)[0]):
    if np.size(np.where(test[i] == np.max(test[i]))) == 1:
        output[i] = np.where(test[i] == np.max(test[i]))[0][0]
    else:
        output[i] = np.NAN
    if output[i] != mnist_test_labels[i]:
        mistakes.append(i)

print(f"grade: {len(mistakes)/np.size(mnist_test_labels)}")

# generate plot of mnist figure at idx
# idx = 7777
# plt.imshow(mnist_train_data[idx],cmap="gray")
# print(mnist_train_labels[idx])