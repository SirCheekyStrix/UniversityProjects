#ifndef BANK_H
#define BANK_H

#include "Account.h"
#include "Transaction.h"

using namespace std;

class Bank {
public:
    Bank();

    Account* createAccount(int accountNumber, double initialBalance);
    Transaction* addTransaction(Account* sender, Account* receiver, double amount);
    void listBalance(Account* account);
    void listTransactionHistory(Account* account);
    Account* accounts[100];  
    int accountCount;        
    Transaction* transactions[100];  
    int transactionCount;            
};
#endif