#include<iostream>
#include<string>
#include "RandomMachine.h"

void menu() {
    std::cout << "\nMenu:\n";
    std::cout << "1. Dodaj item\n";
    std::cout << "2. Wypisz maszyne\n";
    std::cout << "3. Losuj z powtórzeniami\n";
    std::cout << "4. Losuj bez powtórzeń\n";
    std::cout << "5. Wypisz wylosowane przedmioty\n";
    std::cout << "6. Wyjście\n";
    std::cout << "Podaj wybór: ";
}

template <typename T>
void userChoiceHandler(RandomMachine<T>& machine) {
    int choice, count;
    while (1) {
        menu();
        std::cin >> choice;

        switch (choice) {
            case 1: {
                T item;
                std::cout << "Podaj item: ";
                std::cin >> item;
                machine.addItem(item);
                break;
            }
            case 2:
                std::cout << "Itemy w maszynie:\n";
                machine.printItems();
                break;
            case 3:
                std::cout << "Ile wylosować? ";
                std::cin >> count;
                machine.drawSet(count, true);
                break;
            case 4:
                std::cout << "Ile wylosować? ";
                std::cin >> count;
                machine.drawSet(count, false);
                break;
            case 5:
                std::cout << "Wylosowane przedmioty:\n";
                machine.printDrawnItems();
                break;
            case 6:
                std::cout << "Wyjście...\n";
                return;
            default:
                std::cout << "Niepoprawny wybór.\n";
        }
    }
}

void userCustomChoiceHandler(RandomMachine<CustomType>& machine) {
    int choice, count, id;
    std::string name;
    while (1) {
        menu();
        std::cin >> choice;

        switch (choice) {
            case 1:
                std::cout << "Podaj id itemu: ";
                std::cin >> id;
                std::cout << "Podaj item: ";
                std::cin.ignore();
                std::getline(std::cin, name);
                machine.addItem(CustomType(id, name));
                break;
            case 2:
                std::cout << "Itemy w maszynie:\n";
                machine.printItems();
                break;
            case 3:
                std::cout << "Ile wylosować? ";
                std::cin >> count;
                machine.drawSet(count, true);
                break;
            case 4:
                std::cout << "Ile wylosować? ";
                std::cin >> count;
                machine.drawSet(count, false);
                break;
            case 5:
                std::cout << "Wylosowane przedmioty:\n";
                machine.printDrawnItems();
                break;
            case 6:
                std::cout << "Wyjście...\n";
                return;
            default:
                std::cout << "Niepoprawny wybór.\n";
        }
    }
}

int main() {
    int typ;
    std::cout << "Wybierz typ danych:\n";
    std::cout << "1. Integer\n";
    std::cout << "2. Double\n";
    std::cout << "3. String\n";
    std::cout << "4. Custom\n";
    std::cout << "Podaj wybór: ";
    std::cin >> typ;

    if ( typ == 1 ) {
        RandomMachine<int> intMachine;
        userChoiceHandler(intMachine);
    }
    else if ( typ == 2 ) {
        RandomMachine<double> doubleMachine;
        userChoiceHandler(doubleMachine);
    }
    else if ( typ == 3 ) {
        RandomMachine<std::string> stringMachine;
        userChoiceHandler(stringMachine);
    }
    else if ( typ == 4 ) {
        RandomMachine<CustomType> customMachine;
        userCustomChoiceHandler(customMachine);
    }
    else {
        std::cout << "Niepoprawny wybór . . . \n";
        return 1;
    }
    return 0;
}