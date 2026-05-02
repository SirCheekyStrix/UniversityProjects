#ifndef COMPLEX_H
#define COMPLEX_H

#include<iostream>
#include<cmath>

class Complex {
public:
    Complex(); 
    Complex(double real, double imaginary); 

    double getReal() const; 
    double getImaginary() const; 

    void setReal(double real);
    void setImaginary(double imaginary);

    Complex operator+(const Complex& other) const;
    Complex operator-(const Complex& other) const;
    Complex operator*(const Complex& other) const;
    Complex operator/(const Complex& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Complex& complex);
    friend std::istream& operator>>(std::istream& is, Complex& complex);

    bool operator==(const Complex& other) const;
    bool operator!=(const Complex& other) const;

    double modulus() const;

private:
    double real;
    double imaginary;
};

#endif