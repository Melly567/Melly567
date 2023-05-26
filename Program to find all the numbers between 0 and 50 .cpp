#include <iostream>

int main() {
    int sum=0;
    for (int i = 0; i <= 50; i++){
        sum += i;
    }
    std::cout << "The sum of all numbers between 0 and 50 is: " << sum << std::endl;
    return 0;
}
