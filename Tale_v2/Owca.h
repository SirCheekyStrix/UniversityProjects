#ifndef OWCA_H
#define OWCA_H

class Owca {
    public:
        int siarka;
        Owca(int _siarka);
        virtual ~Owca() {}; 
        virtual void makeSound() const;  
};

#endif