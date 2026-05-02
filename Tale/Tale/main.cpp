#include<iostream>
#include<unistd.h>
#include "Smok.h"
#include "Owca.h"
#include "Mieszkaniec.h"

using namespace std;

int main () {
	cout << "Tak dawno temu, że ledwo pamiętam... " << endl;
	Smok s = Smok();
	Mieszkaniec m = Mieszkaniec();	
	m.powitaj(&s);
	s.powitaj(&m);
	Owca o1 = Owca(0);
	Owca o2 = Owca(3);
	Owca o3 = Owca(10);
	Owca o4 = Owca(2);
	Owca o5 = Owca(9);
	cout << "Smok zjadł pierwszą owcę!" << endl;
	s.zjedz(&o1);
	cout << "Smok podpalił mieszkańca!" << endl;
	s.podpal(&m);
	m.plone(&s);
	cout << "Smok zjadł drugą owcę!" << endl;
	s.zjedz(&o2);
	cout << "Smok ugryzł mieszkańca!" << endl;
	s.gryz(&m);
	m.odgryz(&s);
	cout << "Smok zjadł trzecią owcę!" << endl;
	s.zjedz(&o3);
	cout << "Smok zjadł czwartą owcę!" << endl;
	s.zjedz(&o4);
	cout << "Smok podpalił mieszkańca!" << endl;
	s.podpal(&m);
	m.plone(&s);
	cout << "Smok zjadł piątą owcę!" << endl;
	s.zjedz(&o5);
}
