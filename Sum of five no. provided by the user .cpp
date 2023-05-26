#include <iostream>

int main() {
    int sum = 0;
            int number;
    std::cout << "Enter five numbers: " << std::endl;

    for (int i= 1; i<= 5; i++){
        std::cout << "Number" << i << ": ";
        std::cin >> number;
        sum+= number;
    }
    std::cout << "The sum is: "<< sum << std::endl;
    return 0;
}
