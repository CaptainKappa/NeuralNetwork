import numpy
import matplotlib.pyplot as plt
from PIL import Image
from NeuralNetwork import NeuralNetwork

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 10
output_nodes = 10

learningRate = 0.2

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learningRate)

# each entry in CSV-file conists of [label(0-9),0,0,142,69,...,0 (700+)]

# read training_dataset
training_data_file = open("N:\Dataset\mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close

# read test_dataset
test_data_file = open("N:\Dataset\mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close

# train network
for record in training_data_list:
    # .split cuts string whenever it reads a ','
    all_values = record.split(",")
    # scale and shift the inputs
    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create a vector which contains 0.01 for every false label and 0.99 for the right label
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(scaled_input, targets)
    pass

rightPredictions = 0
wrongPredictions = 0

# test network
for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    print(correct_label,"correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # evaluating given entry from test-data
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print(label,"network's answer")
    # updating scoreboard to keep track of its accuracy
    if (label == correct_label):
        rightPredictions += 1
    else:
        wrongPredictions += 1
     
    pass

# own tests
#img_array = numpy.invert(Image.open("C:\\Users\\Marius\\Desktop\\acht.png").convert('L')).ravel()
#print("Network Predicted:",numpy.argmax(n.query(img_array)))

print("Right Predictions:",rightPredictions,"\nWrong Predictions:",wrongPredictions,
      "\nPercentage:",(rightPredictions/len(test_data_list))*100) 
