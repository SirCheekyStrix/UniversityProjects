#ifndef ACCOUNT_H
#define ACCOUNT_H

#include <string>
#include <vector>
#include <iostream>
#include "Transaction.h"

class Transaction;

class Account {
    friend class Transaction;
    friend class Bank;
    friend std::ostream& operator<<(std::ostream& os, const Account& account);

public:
    static Account* createAccount(int accountNumber, double initialBalance);

    int getAccountNumber() const;
    double getBalance() const;
    const std::vector<Transaction*>& getTransactions() const;
    ~Account();

    static std::string formatAccountNumber(int accountNumber);
    static std::string formatAmount(double amount);

private:
    Account(int accountNumber, double initialBalance);

    int accountNumber;
    double balance;
    std::vector<Transaction*> transactions; 
    int transactionCount;

    void addTransaction(Transaction* transaction);
};

std::ostream& operator<<(std::ostream& os, const Account& account);

#endif