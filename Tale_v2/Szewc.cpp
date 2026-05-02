#include "Szewc.h"
#include <iostream>
#include <unistd.h>
using namespace std;

Szewc::Szewc() : Mieszkaniec() {
    cout << "Szewc może już tworzyć nafaszerowanie owce!" << endl;
    sleep(1);
}

OwcaNadziana* Szewc::nadziej_owce(int _siarka) {
    OwcaNadziana* no = new OwcaNadziana(_siarka);
    dodajNadzianaOwce(no);
    return no;
}

