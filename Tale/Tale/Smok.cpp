#include "Smok.h"


void Smok::zjedz(Owca *o){
	zjedzona_siarka += o->siarka;
	cout << "Zjadłem już " << zjedzona_siarka << " kg siarki!" << endl;
	sleep(1);
	if (zjedzona_siarka >= 10 && zjedzona_siarka < 20){
		cout << "Ale chce mi się pić!" << endl;
		sleep(1);
	}
	else if (zjedzona_siarka >= 20){
		cout << "(Pije dużo wody) Padam z wypicia za dużej ilości wody" << endl;
		sleep(1);
	}
}
Smok::Smok(){
	zjedzona_siarka = 0;
	cout << "Nadlatuję aby zjeść trochę waszych owiec..." << endl;
	sleep(1);
}
void Smok::powitaj(Mieszkaniec *m){
	cout << "Witaj mieszkańcu!" << endl;
	sleep(1);
}
void Smok::podpal(Mieszkaniec *m){
	cout << "Podpalę cię mieszkańcu (podmuch ognia) !" << endl;
	sleep(1);
}
void Smok::gryz(Mieszkaniec *m){
	cout << "Gryzę cię mieszkańcu (głośne odgłosy zębów) !" << endl;
	sleep(1);
}
