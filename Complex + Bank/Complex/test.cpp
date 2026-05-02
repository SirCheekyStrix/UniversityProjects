#include "ComplexTest.h"
#include <iostream>

using namespace std;

int main() {
    ComplexTest complexTest;
    int choice;
    
    cout << " Wybierz opcję:" << endl;
    cout << "1. Wszystkie testy" << endl;
    cout << "2. Konkretny test" << endl;
    cout << "Podaj wybór: ";
    cin >> choice;

    if (choice == 1) {
        complexTest.testAll();
    } else if (choice == 2) {
        int testNumber;
        cout << "Podaj numer (1-18): ";
        cin >> testNumber;
        complexTest.runTest(testNumber);
    } else {
        cout << "Zły wybór." << endl;
    }

    return 0;
}
