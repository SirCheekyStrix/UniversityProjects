#include "Smok.h"
#include <iostream>
#include <unistd.h>
using namespace std;

Smok::Smok(Rasa _rasa, int _limitSiarki, int _limitWody)
    : rasa(_rasa), zjedzonaSiarka(0), limitSiarki(_limitSiarki), poziomPragnienia(0), limitWody(_limitWody), zywy(true) {
    cout << "Pojawił się smok rasy " << getRasa() << "!" << endl;
    sleep(1);
}

void Smok::zionie_ogniem() {
    if (!zywy) {
        cout << getRasa() << " nie żyje i nie może zionąć." << endl;
        sleep(1);
	return;
    }

    switch (rasa) {
        case OGNIOSMOK:
            cout << "Smok zionie ogniem!" << endl;
            sleep(1);
	    break;
        case WODNYSMOK:
            cout << "Smok pluje wodą!" << endl;
            sleep(1);
	    break;
        case ZIEMNYSMOK:
            cout << "Smok rzuca kamieniami!" << endl;
            sleep(1);
	    break;
    }
}

void Smok::zjedzOwce(Owca* owca) {
    if (!zywy) {
        cout << getRasa() << " nie żyje i nie może jeść owiec." << endl;
        sleep(1);
	return;
    }

    OwcaNadziana* owcaNadziana = dynamic_cast<OwcaNadziana*>(owca);
    if (owcaNadziana) {
        cout <<  getRasa() << " zjada nafaszerowaną owcę!" << endl;
        sleep(1);
	zjedzonaSiarka += owcaNadziana->siarka;
    } else {
        cout << getRasa() << " zjada zwykłą owcę!" << endl;
        sleep(1);
	zjedzonaSiarka += owca->siarka;
    }

    poziomPragnienia += owca->siarka / 2;

    if (zjedzonaSiarka > limitSiarki) {
        zywy = false;
        cout << getRasa() << " zjadł za dużo siarki i zginął." << endl;
    	sleep(1);
    }
}

void Smok::pijWode(int woda) {
    if (!zywy) {
        cout << getRasa() << " nie żyje i nie może pić wody." << endl;
        sleep(1);
	return;
    }

    if (woda > limitWody) {
        zywy = false;
        cout << getRasa() << " wypił za dużo wody i zginął z przepicia." << endl;
    	sleep(1);
    } else {
        poziomPragnienia -= woda;
        if (poziomPragnienia < 0) {
            poziomPragnienia = 0;
        }
        cout << getRasa() << " wypił " << woda << " litrów wody." << endl;
    	sleep(1);
    }
}

string Smok::getRasa() const {
    switch (rasa) {
        case OGNIOSMOK: return "Ognismok";
        case WODNYSMOK: return "Wodnysmok";
        case ZIEMNYSMOK: return "Ziemnysmok";
        default: return "Nieznana rasa";
    }
}

bool Smok::czyZywy() const {
    return zywy;
}

