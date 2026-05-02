#include "Bank.h"
#include <iostream>

using namespace std;

Bank::Bank() : accountCount(0), transactionCount(0), accountCapacity(10), transactionCapacity(10) {
    accounts = new Account*[accountCapacity];
    transactions = new Transaction*[transactionCapacity];
}

Bank::~Bank() {
    for (int i = 0; i < accountCount; ++i) {
        delete accounts[i];
    }
    delete[] accounts;

    for (int i = 0; i < transactionCount; ++i) {
        delete transactions[i];
    }
    delete[] transactions;
}

void Bank::resizeAccounts() {
    accountCapacity *= 2;
    Account** newAccounts = new Account*[accountCapacity];
    for (int i = 0; i < accountCount; ++i) {
        newAccounts[i] = accounts[i];
    }
    delete[] accounts;
    accounts = newAccounts;
}

void Bank::resizeTransactions() {
    transactionCapacity *= 2;
    Transaction** newTransactions = new Transaction*[transactionCapacity];
    for (int i = 0; i < transactionCount; ++i) {
        newTransactions[i] = transactions[i];
    }
    delete[] transactions;
    transactions = newTransactions;
}

Account* Bank::createAccount(int accountNumber, double initialBalance) {
    if (accountCount == accountCapacity) {
        resizeAccounts();
    }
    Account* account = Account::createAccount(accountNumber, initialBalance);
    accounts[accountCount++] = account;
    return account;
}

Account* Bank::createPublicAccount(int accountNumber, double initialBalance) {
    return createAccount(accountNumber, initialBalance);
}

Transaction* Bank::addTransaction(Account* sender, Account* receiver, double amount) {
    if (transactionCount == transactionCapacity) {
        resizeTransactions();
    }
    Transaction* transaction = Transaction::createTransaction(sender, receiver, amount);
    if (transaction != nullptr) {
        transactions[transactionCount++] = transaction;
    }
    return transaction;
}

Account* Bank::getAccount(int accountNumber) const {
    for (int i = 0; i < accountCount; ++i) {
        if (accounts[i]->getAccountNumber() == accountNumber) {
            return accounts[i];
        }
    }
    return nullptr;
}

void Bank::listBalance(Account* account) const {
    cout << *account;
}

void Bank::listTransactionHistory(Account* account) const {
    cout << "Historia transakcji dla konta o numerze: " << Account::formatAccountNumber(account->getAccountNumber()) << endl;
    for (const auto& t : account->getTransactions()) {
        cout << *t;
    }
}
