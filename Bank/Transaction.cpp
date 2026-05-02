#include "Transaction.h"
#include "Account.h"

Transaction::Transaction(Account* sender, Account* receiver, double amount)
    : sender(sender), receiver(receiver), amount(amount) {
    if (sender->balance >= amount) {
        sender->balance -= amount;
        receiver->balance += amount;
        sender->addTransaction(this);
        receiver->addTransaction(this);
    }
}

Transaction* Transaction::createTransaction(Account* sender, Account* receiver, double amount) {
    return new Transaction(sender, receiver, amount);
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