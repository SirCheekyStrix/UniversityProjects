#include <iostream>
#include "Bank.h"
#include "IOHandler.h"

using namespace std;

int main() {
    Bank bank;
    IOHandler ioHandler;


    Account* account1 = bank.createPublicAccount(1001, 5000.0);
    Account* account2 = bank.createPublicAccount(1002, 3000.0);
    Account* account3 = bank.createPublicAccount(1003, 7000.0);


    bank.addTransaction(account1, account2, 1500.0);
    bank.addTransaction(account2, account3, 2000.0);
    bank.addTransaction(account3, account1, 2500.0);


    ioHandler.listBalance(account1);
    ioHandler.listBalance(account2);
    ioHandler.listBalance(account3);


    ioHandler.listTransactionHistory(account1);
    ioHandler.listTransactionHistory(account2);
    ioHandler.listTransactionHistory(account3);

    return 0;
}
