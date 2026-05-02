#include "Account.h"

Account::Account(int accountNumber, double initialBalance)
    : accountNumber(accountNumber), balance(initialBalance), transactionCount(0) {}
Account* Account::createAccount(int accountNumber, double initialBalance) {
    return new Account(accountNumber, initialBalance);
}
void Account::addTransaction(Transaction* transaction) {
    if (transactionCount < 100) {
        transactions[transactionCount++] = transaction;
    }
}

int Account::getAccountNumber() const {
    return accountNumber;
}

double Account::getBalance() const {
    return balance;
}