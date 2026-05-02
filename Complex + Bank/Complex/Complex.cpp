#include "Complex.h"

Complex::Complex() : real(0), imaginary(0) {}

Complex::Complex(double real, double imaginary) : real(real), imaginary(imaginary) {}

double Complex::getReal() const {
    return real;
}

double Complex::getImaginary() const {
    return imaginary;
}

void Complex::setReal(double real) {
    this->real = real;
}

void Complex::setImaginary(double imaginary) {
    this->imaginary = imaginary;
}

Complex Complex::operator+(const Complex& other) const { 
    return Complex(this->real + other.real, this->imaginary + other.imaginary);
}

Complex Complex::operator-(const Complex& other) const { 
    return Complex(this->real - other.real, this->imaginary - other.imaginary);
}

Complex Complex::operator*(const Complex& other) const { 
    double newReal = (this->real * other.real) - (this->imaginary * other.imaginary); 
    double newImaginary = (this->real * other.imaginary) + (this->imaginary * other.real); 
    return Complex(newReal, newImaginary);
}

Complex Complex::operator/(const Complex& other) const { 
    double denominator = other.real * other.real + other.imaginary * other.imaginary; 
    double newReal = (this->real * other.real + this->imaginary * other.imaginary) / denominator; 
    double newImaginary = (this->imaginary * other.real - this->real * other.imaginary) / denominator; 
    return Complex(newReal, newImaginary);
}

std::ostream& operator<<(std::ostream& os, const Complex& complex) { 
    os << "(" << complex.getReal() << " + " << complex.getImaginary() << "i)"; 
    return os; 
}

std::istream& operator>>(std::istream& is, Complex& complex) { 
    double real, imaginary; 
    is >> real >> imaginary; 
    complex.setReal(real);
    complex.setImaginary(imaginary); 
    return is;
}

bool Complex::operator==(const Complex& other) const { 
    return (this->real == other.real) && (this->imaginary == other.imaginary); 
}

bool Complex::operator!=(const Complex& other) const { 
    return !(*this == other); 
}  

double Complex::modulus() const {
    return std::sqrt(real * real + imaginary * imaginary); 
}