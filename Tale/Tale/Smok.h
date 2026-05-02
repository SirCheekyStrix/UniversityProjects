#include<iostream>
#include<unistd.h>
#include "Owca.h"

using namespace std;

#ifndef SMOK_H
#define SMOK_H

class Mieszkaniec;

class Smok {
	private:
		int zjedzona_siarka;
	public:
		Smok();
		void zjedz(Owca *o);
		void powitaj(Mieszkaniec *m);
		void podpal(Mieszkaniec *m);
		void gryz(Mieszkaniec *m);
};
#endif
