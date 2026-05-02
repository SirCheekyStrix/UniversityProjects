#ifndef TRANSACTION_H
#define TRANSACTION_H

#include "Account.h"

class Account;

class Transaction {
public:
    static Transaction* createTransaction(Account* sender, Account* receiver, double amount);

    double getAmount() const;
    Account* getSender() const;
    Account* getReceiver() const;

private:
    Transaction(Account* sender, Account* receiver, double amount);

    Account* sender;
    Account* receiver;
    double amount;
};

std::ostream& operator<<(std::ostream& os, const Transaction& transaction);

#endif
