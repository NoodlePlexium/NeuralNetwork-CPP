#ifndef RANDOM_H
#define RANDOM_H

#include <cstdlib>
#include <ctime>

class Random {
public:
    // Constructor that initializes the random number generator
    Random() {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }

    // Destructor (cleanup if needed)
    ~Random() {
        // You can add cleanup code here if necessary
    }

    // Generate a random integer between min and max (inclusive)
    static int randInt(int min, int max) {
        return min + std::rand() % (max - min + 1);
    }

    // Generate a random double between min and max
    static double randDouble(double min, double max) {
        double random = static_cast<double>(std::rand()) / RAND_MAX;
        return min + random * (max - min);
    }
};

#endif // RANDOM_H
