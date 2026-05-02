#ifndef ACCOUNT_H
#define ACCOUNT_H

#include <string>
#include "Transaction.h"

class Transaction;

class Account {
    friend class Transaction;
    friend class Bank;
public:
    static Account* createAccount(int accountNumber, double initialBalance);
    int getAccountNumber() const; 
    double getBalance() const;  

private:
    Account(int accountNumber, double initialBalance);
    int accountNumber;
    double balance;
    Transaction* transactions[100]; 
    int transactionCount;           
     void addTransaction(Transaction* transaction);
};
#endif