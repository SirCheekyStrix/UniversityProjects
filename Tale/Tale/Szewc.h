#ifndef SZEWC_H
#define SZEWC_H

#include "Mieszkaniec.h"
#include "OwcaNadziana.h"

class Szewc : public Mieszkaniec {
    public:
        Szewc();
        OwcaNadziana* nadziej_owce(int _siarka);
};

#endif
