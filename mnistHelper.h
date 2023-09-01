#ifndef MNISTHELPER_H
#define MNISTHELPER_H

class MnistHelper {
public:
	static std::vector<int> getImage(int imageIndex, std::string filePath) {
        std::ifstream inputFile(filePath, std::ios::binary);

        if (!inputFile) {
            std::cerr << "Error opening MNIST image file." << std::endl;
            return {};
        }

        inputFile.seekg(16 + imageIndex * 28 * 28); // Skip header and previous images

        std::vector<int> grayscaleValues;
        for (int i = 0; i < 28 * 28; ++i) {
            std::uint8_t pixelValue;
            inputFile.read(reinterpret_cast<char*>(&pixelValue), sizeof(pixelValue));
            grayscaleValues.push_back(static_cast<int>(pixelValue));
        }

        inputFile.close();
        return grayscaleValues;
    }

    static int getLabel(int imageIndex, std::string filePath) {
        std::ifstream labelFile(filePath, std::ios::binary);

        if (!labelFile) {
            std::cerr << "Error opening MNIST label file." << std::endl;
            return -1; // Return an invalid label
        }

        labelFile.seekg(8 + imageIndex); // Skip header and previous labels

        std::uint8_t label;
        labelFile.read(reinterpret_cast<char*>(&label), sizeof(label));

        labelFile.close();
        return static_cast<int>(label);
    }

};


#endif // MNISTHELPER_H