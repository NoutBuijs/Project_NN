from Tools import NeuralNetwork as NN
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
from Tools.interpolationregressionsplines import oneD_polyregression

np.set_printoptions(precision=1)

inputsize = 2
outputsize = 2

activation_function = lambda x: NN.leak_relu(0.01, x)
derivative_activation_function = lambda x: NN.d_leak_relu(0.01, x)

curv = NN.he(2,2,activation_fuction=activation_function,derivative_activation_function=derivative_activation_function,he_bias=False)

samplemax = 100
epochs = 400
datamax = samplemax*epochs

data = datasets.make_circles(int(datamax), noise=0.06)
testdata = np.vstack(data[0])

solution = np.zeros(np.size(data[1]),dtype=object)
solution[np.where(data[1] == 1)] = [np.array([0,1])]
solution[np.where(data[1] != 1)] = [np.array([1,0])]
solution = np.vstack(solution)

learn = 0
lowerbound = 0
upperbound = samplemax
cmax = np.zeros(epochs)
cavg = np.zeros(epochs)
cmin = np.zeros(epochs)
x = np.linspace(0,epochs-1,epochs)

for i in tqdm(range(np.size(solution[:,0]))):
    if i//samplemax == learn:
        c1,c2,c3 = curv.learn(testdata[lowerbound:upperbound,:],solution[lowerbound:upperbound,:])
        cmax[learn] = c1
        cavg[learn] = c2
        cmin[learn] = c3

        learn += 1

        lowerbound = upperbound
        upperbound += samplemax

data2 = datasets.make_circles(1000)
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

plt.subplot(1,2,1)
plt.title("result")
plt.scatter(smallcircle[:,0],smallcircle[:,1],c="black")
plt.scatter(bigcircle[:,0],bigcircle[:,1],c="red")

order = 12
cmax,amax = oneD_polyregression(x,cmax,order=order)
cavg,aavg = oneD_polyregression(x,cavg,order=order)
cmin,amin = oneD_polyregression(x,cmin,order=order)

plt.subplot(1,2,2)
plt.title("cost over epoch")
plt.plot(x,cmax(x),c="red")
plt.plot(x,cavg(x),c="black")
plt.plot(x,cmin(x),c="green")
plt.legend(("cmax","cavg","cmin"))


print(f"perfomance on new data:\n"
      f"accuracy: {np.round(np.size(correctidx)/np.shape(confidentprediction[0])[0],3)*100}%\n"
      f"Loss: {np.round((np.size(input) - (np.size(smallcircle) + np.size(bigcircle)))/np.size(input),4) * 100}%")
plt.show()
