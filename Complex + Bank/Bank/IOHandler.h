#ifndef IOHANDLER_H
#define IOHANDLER_H

#include "Account.h"
#include "Transaction.h"
#include "Bank.h"

class IOHandler {
public:
    void listBalance(const Account* account) const;
    void listTransactionHistory(const Account* account) const;
    void displayMenu() const;
    int getChoice() const;
    int getAccountNumber() const;
    double getAmount() const;
    void getTwoAccountNumbers(int &sender, int &receiver) const;
};

#endif
