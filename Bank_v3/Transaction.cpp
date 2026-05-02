#include "Transaction.h"
#include "Account.h"
#include <iomanip>

Transaction::Transaction(Account* sender, Account* receiver, double amount)
    : sender(sender), receiver(receiver), amount(amount) {
    if (sender->balance >= amount) {
        sender->balance -= amount;
        receiver->balance += amount;
        sender->addTransaction(this);
        receiver->addTransaction(this);
    } else {
        // Obsługa błędu: saldo nadawcy jest zbyt niskie
        this->sender = nullptr;
        this->receiver = nullptr;
        this->amount = 0;
    }
}

Transaction* Transaction::createTransaction(Account* sender, Account* receiver, double amount) {
    if (sender->getBalance() >= amount) {
        return new Transaction(sender, receiver, amount);
    } else {
        return nullptr;  // Transakcja nie może być zrealizowana
    }
}

double Transaction::getAmount() const {
    return amount;
}

Account* Transaction::getSender() const {
    return sender;
}

Account* Transaction::getReceiver() const {
    return receiver;
}

std::ostream& operator<<(std::ostream& os, const Transaction& transaction) {
    if (transaction.getSender() && transaction.getReceiver()) {
        os << "[" << Account::formatAccountNumber(transaction.getSender()->getAccountNumber())
           << " -> " << Account::formatAccountNumber(transaction.getReceiver()->getAccountNumber())
           << "] Kwota: " << Account::formatAmount(transaction.getAmount()) << "\n";
    } else {
        os << "Invalid transaction\n";
    }
    return os;
}
