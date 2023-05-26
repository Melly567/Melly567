#include <iostream>

int main() {
    int numGrades;
    float total = 0.0f;
    float grade;
    std::cout << "Enter the number of grades: ";
    std::cin >> numGrades;

    for (int i = 1; i  <= numGrades;i++){
        std::cout << "Enter grade" << i <<": ";
        std::cin >> grade;
        total += grade;
    }
    float average = total / numGrades;
    std::cout<<"Average grade: "<< average << std::endl;
    return 0;
}
