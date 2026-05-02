#ifndef MIESZKANIEC_H
#define MIESZKANIEC_H

#include <iostream>
#include "Smok.h"
#include "Owca.h"
#include "OwcaNadziana.h"

class Mieszkaniec {
    public:
        Mieszkaniec();
        ~Mieszkaniec();
        void powitaj(Smok *s);
        void plone(Smok *s);
        void odgryz(Smok *s);
        void dodajOwce(Owca* o);
        void dodajNadzianaOwce(OwcaNadziana* no);
        void wyswietlOwce();
        void wyswietlNadzianeOwce();
    private:
        Owca** owce;
        int liczbaOwiec;
        int maxOwce;
        OwcaNadziana** OwceNadziane;
        int liczbaNadzianychOwiec;
        int maxNadzianychOwiec;
        void rozszerzTabliceOwce();
        void rozszerzTabliceOwceNadziane();
};

#endif

