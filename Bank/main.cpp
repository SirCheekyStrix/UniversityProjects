#include <iostream>
#include "Bank.h"

using namespace std;

void showMenu() {
    cout << "1. Utworz konto" << endl;
    cout << "2. Nowa transakcja" << endl;
    cout << "3. Wyswietl balans" << endl;
    cout << "4. Wyswietl historie transakcji" << endl;
    cout << "5. Wyjscie" << endl;
}

int main() {
    Bank bank;
    int choice;
    do {
        showMenu();
        cin >> choice;
        switch (choice) {
            case 1: {
                int accountNumber;
                double initialBalance;
                cout << "Podaj numer konta: ";
                cin >> accountNumber;
                cout << "Podaj balans: ";
                cin >> initialBalance;
                bank.createAccount(accountNumber, initialBalance);
                break;
            }
            case 2: {
                int senderNumber, receiverNumber;
                double amount;
                cout << "Wprowadz numer sendera: ";
                cin >> senderNumber;
                cout << "Wprowadz numer odbiorcy: ";
                cin >> receiverNumber;
                cout << "Wprowadz sume: ";
                cin >> amount;

                Account* sender = nullptr;
                Account* receiver = nullptr;
                for (int i = 0; i < bank.accountCount; ++i) {
                    if (bank.accounts[i]->getAccountNumber() == senderNumber) sender = bank.accounts[i];
                    if (bank.accounts[i]->getAccountNumber() == receiverNumber) receiver = bank.accounts[i];
                }
                if (sender && receiver) {
                    bank.addTransaction(sender, receiver, amount);
                } else {
                    cout << "Zly numer konta" << endl;
                }
                break;
            }
            case 3: {
                int accountNumber;
                cout << "Wprowadz numer konta: ";
                cin >> accountNumber;

                Account* account = nullptr;
                for (int i = 0; i < bank.accountCount; ++i) {
                    if (bank.accounts[i]->getAccountNumber() == accountNumber) account = bank.accounts[i];
                }
                if (account) {
                    bank.listBalance(account);
                } else {
                    cout << "Zly numer konta" << endl;
                }
                break;
            }
            case 4: {
                int accountNumber;
                cout << "Wprowadz numer konta: ";
                cin >> accountNumber;

                Account* account = nullptr;
                for (int i = 0; i < bank.accountCount; ++i) {
                    if (bank.accounts[i]->getAccountNumber() == accountNumber) account = bank.accounts[i];
                }
                if (account) {
                    bank.listTransactionHistory(account);
                } else {
                    cout << "Zly numer konta" << endl;
                }
                break;
            }
            case 5: {
                cout << "Thank you for choosing our bank" << endl;
                break;
            }
            default: {
                cout << "Zly wybor" << endl;
                break;
            }
        }
    } while (choice != 5);

    return 0;
}