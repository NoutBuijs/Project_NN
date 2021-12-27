from Tools import NeuralNetwork as NN
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

np.set_printoptions(precision=1)

activation_function = lambda x: NN.leak_relu(0.01, x)
derivative_activation_function = lambda x: NN.d_leak_relu(0.01, x)

W = np.loadtxt(r"Weights_Biases\circles_W.txt")
B = np.loadtxt(r"Weights_Biases\circles_B.txt")

curv = NN.he(2,2,activation_fuction=activation_function,derivative_activation_function=derivative_activation_function,he_bias=False)

curv.B = B
curv.W = W

data2 = datasets.make_circles(10000, noise=0.06)
input = data2[0]

solution = np.zeros(np.size(data2[1]),dtype=object)
solution[np.where(data2[1] == 1)] = [np.array([0,1])]
solution[np.where(data2[1] != 1)] = [np.array([1,0])]
solution = np.vstack(solution)

cluster = curv.predict(input)
cluster = np.vstack(cluster)

smallcircle = input[np.where(np.all(np.round(cluster) == [0,1],axis=1)),:][0]
bigcircle = input[np.where(np.all(np.round(cluster) == [1,0],axis=1)),:][0]

confidentidx = np.sort(np.hstack((np.where(np.all(np.round(cluster)==[0,1],axis=1)),np.where(np.all(np.round(cluster)==[1,0],axis=1)))))
confidentprediction = cluster[confidentidx]
confidentsolution = solution[confidentidx]
correctidx = np.where(np.all(np.round(confidentprediction)[0] == confidentsolution[0],axis = 1))

plt.subplot(1,1,1)
plt.title("result")
plt.scatter(smallcircle[:,0],smallcircle[:,1],c="black")
plt.scatter(bigcircle[:,0],bigcircle[:,1],c="red")

print(f"perfomance:\n"
      f"accuracy: {np.round(np.size(correctidx)/np.shape(confidentprediction[0])[0],3)*100}%\n"
      f"Loss: {np.round((np.size(input) - (np.size(smallcircle) + np.size(bigcircle)))/np.size(input),4) * 100}%")
plt.show()
