#include "OwcaNadziana.h"
#include <iostream>
#include <unistd.h>
using namespace std;

OwcaNadziana::OwcaNadziana(int _siarka) : Owca(_siarka) {
}

void OwcaNadziana::makeSound() const {
    cout << " (Nadziane) Beeee!" << endl;
}
