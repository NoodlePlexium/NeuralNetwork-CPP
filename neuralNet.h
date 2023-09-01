
#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib> 
#include <algorithm>
#include <omp.h>
#include "random.h"

struct TrainingExample {
    std::vector<double> inputs;
    std::vector<double> expectedOutput;
    int answerIndex;
};


class NeuralNetwork {
private:    
    std::vector<int> structure;
    int layerCount;

    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> neurons;
    std::vector<std::vector<double>> biases;

    void GenerateNetwork() {

        layerCount = structure.size();

        // Gnerate neurons and biases
        for (int i=0; i<layerCount; ++i){

            neurons.push_back(std::vector<double>(structure[i])); // Push back empty neurons
            biases.push_back(std::vector<double>(structure[i]));  // Push back empty biases

            for (int j=0; j<structure[i]; ++j){
                neurons[i][j] = 0;
                biases[i][j] = 0; 
            }
        }


        // Generate weights
        for (int i = 1; i < layerCount; ++i) {
            weights.push_back({});

            for (int in = 0; in < structure[i - 1]; ++in) {
                weights[i - 1].push_back({});

                for (int out = 0; out < structure[i]; ++out) {
                    weights[i - 1][in].push_back(Random::randDouble(-1, 1));
                }
            }
        }

    }

    double sigmoid(double x){
        x = std::max(-500.0, std::min(500.0, x));
        return 1.0/(1.0 + exp(-x));
    }

    double sigmoidDerivative(double x){
        double activation = sigmoid(x);
        return activation * (1 - activation);
    }

    double neuronCost(double predicted, double expected){
        double error = predicted - expected;
        return error * error;
    }

    double neuronCostDerivative(double weightedInput, double expected){
        return 2 * (weightedInput - expected);
    }

    int getPredictionIndex() {

        // Find the index of the maximum activation in the predicted output
        int predictedAnswerIndex = 0;
        double maxPredictedActivation = 0.0;

        for (int i=0; i<neurons[structure.size()-1].size(); ++i) {
            if (neurons[neurons.size()-1][i] > maxPredictedActivation) {
                maxPredictedActivation = neurons[structure.size()-1][i];
                predictedAnswerIndex = i;
            }
        }

        return predictedAnswerIndex;
    }

    void drawProgressBar(int current, int total, int width){
        const int barWidth = 50;
        float progress = static_cast<float>(current) / total;
        int progressBarLength = static_cast<int>(progress * barWidth);

        std::cout << "[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < progressBarLength) {
                std::cout << "=";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << int(progress * 100.0) << "%\r";
        std::cout.flush();
    }

public:
    NeuralNetwork(std::vector<int> _structure) : structure(_structure) { GenerateNetwork(); }   

    void SetInputs(std::vector<double> inputVector) {
        neurons[0] = inputVector;
    }

    void PrintLastLayer() {
        for (int n=0; n<structure[structure.size()-1]; ++n){
            std::cout << neurons[neurons.size()-1][n] << "    " << std::endl;
        }
    }

    void FeedForward() {
        for (int i=1; i<layerCount; ++i) {
            for (int out=0; out<structure[i]; ++out){


                // sum = Σ of (each input neuron * weight connection)
                double sum = 0.0;
                for (int in=0; in<structure[i-1]; ++in){
                    sum += weights[i-1][in][out] * neurons[i-1][in];
                }

                // out neuron's activation = signmoid(sum + bias)
                neurons[i][out] = sigmoid(sum + biases[i][out]);
            }
        }
    }

    void BackPropagate(std::vector<TrainingExample> batch, double learnRate){

        std::vector<std::vector<std::vector<double>>> newWeights = weights;
        std::vector<std::vector<double>> newBiases = biases;

        // Track accuracy
        int numCorrect = 0;
        int completedExamples = 0;


        for (const TrainingExample& example : batch) {


            // Set activations of first layer and feed forward
            SetInputs(example.inputs);
            FeedForward();


            // got the answer correct
            if (getPredictionIndex() == example.answerIndex) numCorrect++;


            // Iterate backwards through the network
            std::vector<double> nodeValues;


            // LAST LAYER ///////////////////////////
            for (int out=0; out<structure[layerCount-1]; ++out){

                int i=layerCount-1;

                // a and node values
                double a = neurons[i][out];
                nodeValues.push_back(neuronCostDerivative(a, example.expectedOutput[out]) * sigmoidDerivative(a));
                
                // Calculate weight delta
                for (int in=0; in<structure[i-1]; ++in){
                    double weightChange = learnRate * neurons[i-1][in] * nodeValues[out];
                    newWeights[i-1][in][out] -= weightChange / batch.size();
                }

                // Calculate bias delta
                double biasChange = learnRate * nodeValues[out];
                newBiases[i][out] -= biasChange / batch.size();
            }


            // HIDDEN LAYERS ////////////////////////
            for (int i=structure.size()-2; i>0; --i){

                std::vector<double> newNodeValues;

                // for out neuron 
                for (int out=0; out<structure[i]; ++out){
                    double outWeightTotal = 0;
                    double weightedInput = 0;

                    // Calculate new node values
                    for (int next=0; next<structure[i+1]; ++next) { outWeightTotal += weights[i][out][next] * nodeValues[next]; }
                    for (int in=0; in<structure[i-1]; ++in)       { weightedInput += weights[i-1][in][out]; }
                    newNodeValues.push_back(outWeightTotal * sigmoidDerivative(weightedInput));


                    // Calculate bias deltas
                    double biasChange = learnRate * newNodeValues[out];
                    newBiases[i][out] -= biasChange / batch.size();

                    // Calculate weight deltas
                    for (int in=0; in<structure[i-1]; ++in){
                        double weightChange = learnRate * newNodeValues[out] * neurons[i-1][in];
                        newWeights[i-1][in][out] -= weightChange / batch.size();
                    }
                }
                nodeValues = newNodeValues;
            }

            // Print progress bar
            completedExamples += 1;
            drawProgressBar(completedExamples, batch.size(), 20);
        }

        // Update all weights and biases
        weights = newWeights;
        biases = newBiases;

        // New line after progress bar
        std::cout << std::endl;
        std::cout << "Accuracy: " << (static_cast<double>(numCorrect) / batch.size()) * 100 << "%" << std::endl;
    }

    void SaveBrain(const std::string& filePath) {

        std::cout << "[Saving Neural Network ...]";

        std::ofstream outFile(filePath);
        if (!outFile.is_open()) {
            std::cerr << "Error opening file for saving weights and biases: " << filePath << std::endl;
            return;
        }

        // Save weights 
        for (int i=1; i<layerCount; ++i){
            for (int out=0; out<structure[i]; ++out){
                for (int in=0; in<structure[i-1]; ++in){
                    outFile << i-1 << " " << in << " " << out << " " << weights[i-1][in][out] << "\n";
                }
            }
        } 


        // Save biases
        for (int i=1; i<layerCount; ++i) {
            for (int j=0; j<biases[i].size(); ++j) {
                outFile << "bias " << i << " " << j << " " << biases[i][j] << "\n";
            }
        }

        outFile.close();

        std::cout << "[Saved]";
    }

    void LoadBrain(const std::string& filePath) {

        std::cout << "[Loading Neural Network ...]";


        std::ifstream inFile(filePath);
        if (!inFile.is_open()) {
            std::cerr << "Error opening file for loading weights and biases: " << filePath << std::endl;
            return;
        }

        // Load weights and biases from the file
        std::string line;
        std::string entryType;
        while (std::getline(inFile, line)) {

            int one, two, three;
            double value;

            std::istringstream iss(line);
            iss >> one >> two >> three >> value;


            if (entryType == "bias") 
            {
                biases[two][three] = value;
            } 
            else 
            {
                weights[one][two][three] = value;
            }
        }

        inFile.close();
        std::cout << "[Complete]";
    }

};

#endif 