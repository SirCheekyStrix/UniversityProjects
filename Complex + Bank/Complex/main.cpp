#include <iostream>
#include "Complex.h"

using namespace std;

int main() {
    Complex c1;
    Complex c2;
    cout << "Podaj c1: " << endl;
    cin >> c1;
    cout << "Podaj c2: " << endl;
    cin >> c2;

    Complex c3 = c1 + c2;
    Complex c4 = c1 - c2;
    Complex c5 = c1 * c2;
    Complex c6 = c1 / c2;

    cout << "c1: " << c1 << endl;
    cout << "c2: " << c2 << endl;
    cout << "c1 + c2 = " << c3 << endl;
    cout << "c1 - c2 = " << c4 << endl;
    cout << "c1 * c2 = " << c5 << endl;
    cout << "c1 / c2 = " << c6 << endl;

    cout << "Moduł  c1: " << c1.modulus() << endl;
    cout << "Moduł  c2: " << c2.modulus() << endl;

    cout << "c1 == c2: " << (c1 == c2 ? "true" : "false") << endl;
    cout << "c1 == c3: " << (c1 == c3 ? "true" : "false") << endl;
    cout << "c1 != c2: " << (c1 != c2 ? "true" : "false") << endl;
    cout << "c1 != c3: " << (c1 != c3 ? "true" : "false") << endl;

    return 0;
}