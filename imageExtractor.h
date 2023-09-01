#ifndef IMAGEEXTRACTOR_H
#define IMAGEEXTRACTOR_H

#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <array>

struct Pixel {
    int r;
    int g;
    int b;
};



class ImageExtractor {
public:
    static std::vector<Pixel> getRGB(const std::string& imagePath) {
        std::ifstream inputFile(imagePath, std::ios::binary);

        if (!inputFile) {
            std::cerr << "Error opening image: " << imagePath << std::endl;
            return {};
        }

        // Seek to the position of the image data
        inputFile.seekg(0x12, std::ios::beg);

        std::vector<Pixel> pixelValues;

        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                std::array<std::uint8_t, 3> pixel;
                inputFile.read(reinterpret_cast<char*>(pixel.data()), pixel.size());

                Pixel p;
                p.r = pixel[0];
                p.g = pixel[1];
                p.b = pixel[2];
                
                pixelValues.push_back(p);
            }
        }

        inputFile.close();
        return pixelValues;
    }

    static std::vector<int> getMNISTImage(int imageIndex, std::string filePath) {
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

private:
    static constexpr int WIDTH = 64; // Adjust according to your image dimensions
    static constexpr int HEIGHT = 36; // Adjust according to your image dimensions
};

#endif // IMAGEEXTRACTOR_H



