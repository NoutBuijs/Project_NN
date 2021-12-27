import torchvision
from Tools import NeuralNetwork as NN
import numpy as np
from sklearn.preprocessing import normalize
np.set_printoptions(precision=3)

mnist_test = torchvision.datasets.MNIST(r"mnist_data",train=False)
mnist_test_data = mnist_test.test_data.numpy()
mnist_test_labels = mnist_test.test_labels.numpy()
shape_test = (np.shape(mnist_test_data)[0],np.shape(mnist_test_data)[1]**2)
mnist_test_data = np.reshape(mnist_test_data,shape_test)
mnist_test_data = normalize(mnist_test_data,norm="max")

nn = NN.he(np.shape(mnist_test_data)[1],10,
           activation_fuction=lambda x:NN.leak_relu(0.01, x),
           derivative_activation_function=lambda x:NN.d_leak_relu(0.01, x))

W = np.loadtxt(r"Weights_Biases\Mnist_W.txt")
B = np.loadtxt(r"Weights_Biases\Mnist_B.txt")

nn.W = W
nn.B = B

inputreductionfactortest = 0.10
maxinput = int(np.ceil(np.shape(mnist_test_data)[0]))
test = nn.predict_tqdm(mnist_test_data[:maxinput])

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