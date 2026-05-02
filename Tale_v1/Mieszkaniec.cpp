#include <unistd.h>
#include "Mieszkaniec.h"
using namespace std;

Mieszkaniec::Mieszkaniec() : liczbaOwiec(0), maxOwce(50), liczbaNadzianychOwiec(0), maxNadzianychOwiec(10) {
    cout << "Jestem mieszkańcem!" << endl;
    owce = new Owca*[maxOwce];
    OwceNadziane = new OwcaNadziana*[maxNadzianychOwiec];
    sleep(1);
}

Mieszkaniec::~Mieszkaniec() {
    for (int i = 0; i < liczbaOwiec; ++i) {
        delete owce[i];
    }
    delete[] owce;
    for (int i = 0; i < liczbaNadzianychOwiec; ++i) {
        delete OwceNadziane[i];
    }
    delete[] OwceNadziane;
}

void Mieszkaniec::powitaj(Smok *s) {
    cout << "Witaj smoku!" << endl;
    sleep(1);
}

void Mieszkaniec::plone(Smok *s) {
    cout << "O nieeeeeeeeeeee!" << endl << "Ja płonę!" << endl;
    sleep(1);
}

void Mieszkaniec::odgryz(Smok *s) {
    cout << "Ugryzł mnie!" << endl << "Uciekam ile sił!" << endl;
    sleep(1);
}

void Mieszkaniec::dodajOwce(Owca* o) {
    if (liczbaOwiec >= maxOwce) {
        rozszerzTabliceOwce();
    }
    owce[liczbaOwiec++] = o;
}

void Mieszkaniec::dodajNadzianaOwce(OwcaNadziana* no) {
    if (liczbaNadzianychOwiec >= maxNadzianychOwiec) {
        rozszerzTabliceOwceNadziane();
    }
    OwceNadziane[liczbaNadzianychOwiec++] = no;
}

void Mieszkaniec::wyswietlOwce() {
    for (int i = 0; i < liczbaOwiec; ++i) {
        cout << "Owca z siarką: " << owce[i]->siarka << endl;
    	sleep(1);
    }
}

void Mieszkaniec::wyswietlNadzianeOwce() {
    for (int i = 0; i < liczbaNadzianychOwiec; ++i) {
        cout << "Nafaszerowana owca z siarką: " << OwceNadziane[i]->siarka << endl;
   	sleep(1);
    }
}

void Mieszkaniec::rozszerzTabliceOwce() {
    maxOwce *= 2;
    Owca** nowaTablica = new Owca*[maxOwce];
    for (int i = 0; i < liczbaOwiec; ++i) {
        nowaTablica[i] = owce[i];
    }
    delete[] owce;
    owce = nowaTablica;
}

void Mieszkaniec::rozszerzTabliceOwceNadziane() {
    maxNadzianychOwiec *= 2;
    OwcaNadziana** nowaTablica = new OwcaNadziana*[maxNadzianychOwiec];
    for (int i = 0; i < liczbaNadzianychOwiec; ++i) {
        nowaTablica[i] = OwceNadziane[i];
    }
    delete[] OwceNadziane;
    OwceNadziane = nowaTablica;
}

