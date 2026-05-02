#include "Bank.h"
#include <iostream>

using namespace std;

Bank::Bank() : accountCount(0), transactionCount(0) {}

Account* Bank::createAccount(int accountNumber, double initialBalance) {
    if (accountCount < 100) {
        Account* account = Account::createAccount(accountNumber, initialBalance);
        accounts[accountCount++] = account;
        return account;
    }
    return nullptr;
}

Transaction* Bank::addTransaction(Account* sender, Account* receiver, double amount) {
    if (transactionCount < 100) {
        Transaction* transaction = Transaction::createTransaction(sender, receiver, amount);
        transactions[transactionCount++] = transaction;
        return transaction;
    }
    return nullptr;
}

void Bank::listBalance(Account* account) {
    cout << "Account number: " << account->getAccountNumber() << " has balance: " << account->getBalance() << endl;
}


void Bank::listTransactionHistory(Account* account) {
    cout << "Transaction history for account number: " << account->getAccountNumber() << endl;
    for (int i = 0; i < account->transactionCount; ++i) {
        Transaction* t = account->transactions[i];
        cout << "Transaction: " << t->getAmount() << " from account " << t->getSender()->getAccountNumber() << " to account " << t->getReceiver()->getAccountNumber() << endl;
    }
}