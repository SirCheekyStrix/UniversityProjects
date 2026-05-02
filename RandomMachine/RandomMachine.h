#ifndef RANDOMMACHINE_H
#define RANDOMMACHINE_H

#include<iostream>
#include<ctime>
#include<string>

const int MAX_CAPACITY = 100;

class CustomType {
public:
    CustomType() : id(0), name("") {}
    CustomType(int id, const std::string& name) : id(id), name(name) {}
    friend std::ostream& operator<<(std::ostream& os, const CustomType& ct) {
        os << "CustomType{id: " << ct.id << ", name: " << ct.name << "}";
        return os;
    }
private:
    int id;
    std::string name;
};

template <typename T>
class RandomMachine {
public:
    RandomMachine();
    ~RandomMachine();
    void addItem(const T& item);
    void printItems() const;
    T* drawSet(int count, bool allowRepeats);
    void printDrawnItems() const;

    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const RandomMachine<U>& rm);

private:
    T items[MAX_CAPACITY];
    T drawnItems[MAX_CAPACITY];
    int size;
    int capacity;
    int drawnSize;

//    void resize();
};

template <typename T>
RandomMachine<T>::RandomMachine() : size(0), capacity(MAX_CAPACITY) {}

template<typename T>
RandomMachine<T>::~RandomMachine() {}

/*template <typename T>
void RandomMachine<T>::resize() {
    capacity *= 2;
    T* newItems = new T[capacity];
    for ( int i = 0; i < size; ++i ) {
        newItems[i] = items[i];
    }
    delete[] items;
    items = newItems;
}
*/
template<typename T>
void RandomMachine<T>::addItem(const T& item) {
    if( size == capacity ) {
        std::cout << "Maksymalna pojemność maszyny!\n";
    }
    items[size++] = item;
}

template <typename T>
void RandomMachine<T>::printItems() const {
    for ( int i = 0; i < size; ++i ) {
        std::cout << items[i] << std::endl;
    }
}

template <typename T>
T* RandomMachine<T>::drawSet(int count, bool allowRepeats) {
    if ( count <= 0 ) {
        std::cout << "Błąd licznika!\n";
        return nullptr;
    }
    if( !allowRepeats && count > size ) {
        std::cout << "Licznik nie może być większy niżliczba elemantów!\n";
        return nullptr;
    }

    drawnSize = count;
    std::srand(std::time(0));

    if (allowRepeats) {
        for ( int i = 0; i < count; ++i ) {
            drawnItems[i] = items[std::rand() % size];
        }
    }
    else {
        bool* select = new  bool[size] { false };
        for (int i = 0; i < count; ++i ) {
            int index;
            do {
                index = std::rand() % size;
            }
            while (select[index]);
            drawnItems[i] = items[index];
            select[index] = true;
        }
        delete[] select;
    }
    return drawnItems;
}
template <typename U>
std::ostream& operator<<(std::ostream& os, const RandomMachine<U>& rm) {
    for ( int i = 0; i < rm.size; ++i ) {
        os << rm.items[i] << " ";
    }
    return os;
}

template <typename T>
void RandomMachine<T>::printDrawnItems() const {
    for ( int i = 0; i < drawnSize; ++i ) {
        std::cout << drawnItems[i] << std::endl;
    }
}
#endif