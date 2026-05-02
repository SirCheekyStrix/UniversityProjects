#include "Account.h"
#include <iomanip>
#include <sstream>

Account::Account(int accountNumber, double initialBalance)
    : accountNumber(accountNumber), balance(initialBalance), transactionCount(0) {}

Account* Account::createAccount(int accountNumber, double initialBalance) {
    return new Account(accountNumber, initialBalance);
}

void Account::addTransaction(Transaction* transaction) {
    transactions.push_back(transaction);
}

int Account::getAccountNumber() const {
    return accountNumber;
}

double Account::getBalance() const {
    return balance;
}

const std::vector<Transaction*>& Account::getTransactions() const {
    return transactions;
}

Account::~Account() {
    for (Transaction* transaction : transactions) {
        delete transaction;
    }
}

std::string Account::formatAccountNumber(int accountNumber) {
    std::ostringstream oss;
    oss << std::setw(10) << std::setfill('0') << accountNumber;
    return oss.str();
}

std::string Account::formatAmount(double amount) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << std::setw(10) << amount;
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Account& account) {
    os << "Konto o numerze: " << Account::formatAccountNumber(account.getAccountNumber()) << "\n"
       << "Balans: " << Account::formatAmount(account.getBalance()) << "\n";
    return os;
}