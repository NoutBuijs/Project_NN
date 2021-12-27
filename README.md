# Project_NN
Creation of a neural network form scratch by me as an introduction to the world of AI.

As a heads up for the reader I started this project not with the intent of creating a state 
of the art neural network; the classes and functions in the tools folder form by no means 
a perfectly designed AI. Lastly, the fundamentals of the network, found in 
"Tools\NeuralNetwork.py", have been retrieved from the content of 3Blue1Brown's lessons on 
neural networks.
 
I worked on this project to put my knowledge to the test and see what I could do, not to push 
the boundaries. Furthermore, working on this project also led to new insights such as the 
effects and relations between different activation functions and initializations on the 
convergence of the network. The "NeuralNetwork.py" file in the tools folder provides a couple 
different initialization schemes and activation functions to play around with, which I would 
highly recommend as it was very enjoyable. Needless to say, I am happy with the results. 

Currently, I have two test in the directory that show the network in action The Circles variant 
is the first test I made for the network and relatively simple, the network is tasked with 
correctly labeling a smaller and larger circle. Next, the mnist variant applies the network on 
the notorious mnist set. The "_trainer" files are able to generate the necessary weights and 
biases. In the case you do not want to go to the trouble of finding these yourself the 
"_application" files use already found values and directly show the performance on the tests.