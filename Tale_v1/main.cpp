#include <iostream>
#include "Szewc.h"
#include "Smok.h"
#include "OwcaNadziana.h"

using namespace std;

int main() {

    cout << "Bajka się zaczyna..." << endl;

    Smok ognismok(Smok::OGNIOSMOK, 50, 20);
    Smok wodnysmok(Smok::WODNYSMOK, 40, 30);
    Smok ziemnysmok(Smok::ZIEMNYSMOK, 60, 25);
    Szewc szewc;
    Mieszkaniec mieszkaniec;

    szewc.powitaj(&ognismok);
    szewc.plone(&ognismok);
    szewc.odgryz(&ognismok);

    szewc.powitaj(&wodnysmok);
    szewc.plone(&wodnysmok);

    szewc.powitaj(&ziemnysmok);
    szewc.odgryz(&ziemnysmok);

    mieszkaniec.powitaj();

    Owca* owca1 = new Owca(10);
    Owca* owca2 = new Owca(15);
    szewc.dodajOwce(owca1);
    szewc.dodajOwce(owca2);

    OwcaNadziana* nafaszerowanaOwca1 = szewc.nadziej_owce(200);
    OwcaNadziana* nafaszerowanaOwca2 = szewc.nadziej_owce(25);

    szewc.wyswietlOwce();
    szewc.wyswietlNadzianeOwce();

    ognismok.zjedzOwce(owca1);
    ognismok.zjedzOwce(nafaszerowanaOwca1);

    wodnysmok.zjedzOwce(owca2);
    wodnysmok.zjedzOwce(nafaszerowanaOwca2);

    ognismok.pijWode(10);
    wodnysmok.pijWode(5);
    ziemnysmok.pijWode(15);
    ziemnysmok.pijWode(1000);
    

    ognismok.zionie_ogniem();
    wodnysmok.zionie_ogniem();
    ziemnysmok.zionie_ogniem();

    return 0;
}

