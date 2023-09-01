#include <iostream>
#include <string>
#include "random.h"
#include "neuralNet.h"
#include "imageExtractor.h"
#include "mnistHelper.h"






// MNIST
std::vector<TrainingExample> CreateMNISTTrainingSet() {
    std::cout << "[Loading MNIST Training Set ...]";
    std::vector<TrainingExample> batch;
    for (int i=0; i<60000; ++i){

        TrainingExample example;

        // get image and label
        std::vector<int> greyValues = MnistHelper::getImage(i, "MNIST/train-images.idx3-ubyte");
        int label = MnistHelper::getLabel(i, "MNIST/train-labels.idx1-ubyte");

        // set inputs to greyscale values
        for (int i=0; i<greyValues.size(); i++){
            example.inputs.push_back(greyValues[i]);
        }

        // set expected output
        example.expectedOutput = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        example.expectedOutput[label] = 1;
        example.answerIndex = label;


        batch.push_back(example);
    }

    std::cout << "[Complete]" << std::endl;
    return batch;
}


int main() {
    // Create a training set
    std::vector<TrainingExample> batch = CreateMNISTTrainingSet();

    // Create a network
    std::vector<int> networkStructure = {784, 50, 20, 10};
    NeuralNetwork network(networkStructure);

    // Load brain
    network.LoadBrain("Brain States/MNIST_brain_784_50_20_10.txt");


    // Train the network
    std::cout << std::endl;
    std::cout << "[Training Network ...]" << std::endl;
    for (int i = 0; i < 10000; ++i) {
        std::cout << " " << std::endl;
        std::cout << "Epoch " << i + 1 << std::endl;

        // Backpropagate with the loaded brain state
        network.BackPropagate(batch, 0.5);
        network.PrintLastLayer();

        // Save network brain
        network.SaveBrain("Brain States/MNIST_brain_784_50_20_10.txt");
        std::cout << " " << std::endl;
    }

    return 0;
}
