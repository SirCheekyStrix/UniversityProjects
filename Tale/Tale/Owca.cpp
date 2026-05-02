#include<iostream>
#include<unistd.h>
#include "Owca.h"

using namespace std;

Owca::Owca(int _siarka){
	cout << "Jestem w stadzie z siarką " << _siarka << " (beee) !" << endl;
	sleep(1);
	siarka = _siarka;
}
