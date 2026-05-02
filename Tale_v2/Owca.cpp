#include "Owca.h"
#include <iostream>
#include <unistd.h>
using namespace std;

Owca::Owca(int _siarka) : siarka(_siarka) {
}

void Owca::makeSound() const {
    cout << "Beeee  !" << endl;
}
