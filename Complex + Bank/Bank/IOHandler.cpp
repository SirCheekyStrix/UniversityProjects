#include "IOHandler.h"
#include <iostream>

using namespace std;

void IOHandler::listBalance(const Account* account) const {
    cout << *account;
}

void IOHandler::listTransactionHistory(const Account* account) const {
    cout << "Historia transakcji dla konta o numerze: " << account->getAccountNumber() << endl;
    for (const auto& t : account->getTransactions()) {
        cout << *t;
    }
}

void IOHandler::displayMenu() const {
    cout << "1. Utwórz konto" << endl;
    cout << "2. Dodaj transakcję" << endl;
    cout << "3. Wyświetl balans" << endl;
    cout << "4. Wyświetl historię transakcji" << endl;
    cout << "5. Wyjście" << endl;
    cout << "Wprowadź wybór: ";
}

int IOHandler::getChoice() const {
    int choice;
    cin >> choice;
    return choice;
}

int IOHandler::getAccountNumber() const {
    int accountNumber;
    cout << "Podaj numer konta: ";
    cin >> accountNumber;
    return accountNumber;
}

double IOHandler::getAmount() const {
    double amount;
    cout << "Podaj stan konta: ";
    cin >> amount;
    return amount;
}

void IOHandler::getTwoAccountNumbers(int &sender, int &receiver) const {
    cout << "Podaj numer wysylajacego: ";
    cin >> sender;
    cout << "Podaj numer odbiorcy: ";
    cin >> receiver;
}
