#include <iostream>
#include "Bank.h"
#include "IOHandler.h"

using namespace std;

int main() {
    Bank bank;
    IOHandler ioHandler;
    int choice;

    while (true) {
        ioHandler.displayMenu();
        choice = ioHandler.getChoice();

        if (choice == 1) {
            int accountNumber = ioHandler.getAccountNumber();
            double initialBalance = ioHandler.getAmount();
            bank.createPublicAccount(accountNumber, initialBalance);
        } else if (choice == 2) {
            int sender, receiver;
            double amount;
            ioHandler.getTwoAccountNumbers(sender, receiver);
            amount = ioHandler.getAmount();
            bank.addTransaction(bank.getAccount(sender), bank.getAccount(receiver), amount);
        } else if (choice == 3) {
            int accountNumber = ioHandler.getAccountNumber();
            Account* account = bank.getAccount(accountNumber);
            ioHandler.listBalance(account);
        } else if (choice == 4) {
            int accountNumber = ioHandler.getAccountNumber();
            Account* account = bank.getAccount(accountNumber);
            ioHandler.listTransactionHistory(account);
        } else if (choice == 5) {
            break;
        } else {
            cout << "Niepoprawny wybór. Spróbuj ponownie." << endl;
        }
    }

    return 0;
}
