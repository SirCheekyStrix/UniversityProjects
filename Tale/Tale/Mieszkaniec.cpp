#include<unistd.h>
#include "Mieszkaniec.h"


void Mieszkaniec::powitaj(Smok *s){
	cout << "Witaj smoku!" << endl;
	sleep(1);
}
void Mieszkaniec::plone(Smok *s){
	cout << "O nieeeeeeeeeeee!" << endl << "Ja płonę!" << endl;
	sleep(1);
}
void Mieszkaniec::odgryz(Smok *s){
	cout << "Ugryzł mnie!" << endl << "Uciekam ile sił!" << endl;
	sleep(1);
}
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
