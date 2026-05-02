#ifndef BANK_H
#define BANK_H

#include "Account.h"
#include "Transaction.h"

class Bank {
public:
    Bank();
    ~Bank();

    Transaction* addTransaction(Account* sender, Account* receiver, double amount);
    Account* getAccount(int accountNumber) const;
    void listBalance(Account* account) const;
    void listTransactionHistory(Account* account) const;

    Account* createPublicAccount(int accountNumber, double initialBalance);

private:
    Account* createAccount(int accountNumber, double initialBalance);
    Account** accounts;
    Transaction** transactions;
    int accountCount;
    int transactionCount;
    int accountCapacity;
    int transactionCapacity;

    void resizeAccounts();
    void resizeTransactions();
};

#endif
