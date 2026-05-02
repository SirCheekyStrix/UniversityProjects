#ifndef OWCA_NADZIANA_H
#define OWCA_NADZIANA_H

#include "Owca.h"

class OwcaNadziana : public Owca {
    public:
        OwcaNadziana(int _siarka);
        ~OwcaNadziana() override {};
        void makeSound() const override; 
};

#endif
