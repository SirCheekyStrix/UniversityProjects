#include "ComplexTest.h"
#include <iostream>

using namespace std;

void ComplexTest::testAll() {
    testAdding1();
    testAdding2();
    testAdding3();
    testSubtracting1();
    testSubtracting2();
    testSubtracting3();
    testMultiplying1();
    testMultiplying2();
    testMultiplying3();
    testDividing1();
    testDividing2();
    testDividing3();
    testEquality1();
    testEquality2();
    testEquality3();
    testModulus1();
    testModulus2();
    testModulus3();

    cout << "Tests passed: " << test_passed << endl;
    cout << "Tests failed: " << test_failed << endl;
}

void ComplexTest::runTest(int testNumber) { 
    switch (testNumber) { 
        case 1: testAdding1(); break; 
        case 2: testAdding2(); break; 
        case 3: testAdding3(); break; 
        case 4: testSubtracting1(); break; 
        case 5: testSubtracting2(); break; 
        case 6: testSubtracting3(); break; 
        case 7: testMultiplying1(); break; 
        case 8: testMultiplying2(); break; 
        case 9: testMultiplying3(); break; 
        case 10: testDividing1(); break; 
        case 11: testDividing2(); break; 
        case 12: testDividing3(); break; 
        case 13: testEquality1(); break; 
        case 14: testEquality2(); break;    
        case 15: testEquality3(); break; 
        case 16: testModulus1(); break; 
        case 17: testModulus2(); break; 
        case 18: testModulus3(); break; 
        default: cout << "Zły numer testu." << endl; break; 
    }

    cout << "Tests passed: " << test_passed << endl;
    cout << "Tests failed: " << test_failed << endl;
}

void ComplexTest::checkTest(const std::string& testName, bool condition) {
    if (condition) {
        cout << testName << ": true" << endl;
        test_passed++;
    } else {
        cout << testName << ": false" << endl;
        test_failed++;
    }
}

void ComplexTest::testAdding1() {
    Complex c1(1.0, 2.0);
    Complex c2(3.0, 4.0);
    Complex result = c1 + c2;
    checkTest("testAdding1", result == Complex(4.0, 6.0));
}

void ComplexTest::testAdding2() {
    Complex c1(-1.0, 2.0);
    Complex c2(3.0, -4.0);
    Complex result = c1 + c2;
    checkTest("testAdding2", result == Complex(2.0, -2.0));
}

void ComplexTest::testAdding3() {
    Complex c1(0.0, 0.0);
    Complex c2(0.0, 0.0);
    Complex result = c1 + c2;
    checkTest("testAdding3", result == Complex(0.0, 0.0));
}

void ComplexTest::testSubtracting1() {
    Complex c1(1.0, 2.0);
    Complex c2(3.0, 4.0);
    Complex result = c1 - c2;
    checkTest("testSubtracting1", result == Complex(-2.0, -2.0));
}

void ComplexTest::testSubtracting2() {
    Complex c1(-1.0, 2.0);
    Complex c2(3.0, -4.0);
    Complex result = c1 - c2;
    checkTest("testSubtracting2", result == Complex(-4.0, 6.0));
}

void ComplexTest::testSubtracting3() {
    Complex c1(0.0, 0.0);
    Complex c2(0.0, 0.0);
    Complex result = c1 - c2;
    checkTest("testSubtracting3", result == Complex(0.0, 0.0));
}

void ComplexTest::testMultiplying1() {
    Complex c1(1.0, 2.0);
    Complex c2(3.0, 4.0);
    Complex result = c1 * c2;
    checkTest("testMultiplying1", result == Complex(-5.0, 10.0));
}

void ComplexTest::testMultiplying2() {
    Complex c1(-1.0, 2.0);
    Complex c2(3.0, -4.0);
    Complex result = c1 * c2;
    checkTest("testMultiplying2", result == Complex(5.0, 10.0));
}

void ComplexTest::testMultiplying3() {
    Complex c1(0.0, 0.0);
    Complex c2(3.0, 4.0);
    Complex result = c1 * c2;
    checkTest("testMultiplying3", result == Complex(0.0, 0.0));
}

void ComplexTest::testDividing1() {
    Complex c1(1.0, 2.0);
    Complex c2(3.0, 4.0);
    Complex result = c1 / c2;
    checkTest("testDividing1", result == Complex(0.44, 0.08));
}

void ComplexTest::testDividing2() {
    Complex c1(-1.0, 2.0);
    Complex c2(3.0, -4.0);
    Complex result = c1 / c2;
    checkTest("testDividing2", result == Complex(-0.44, 0.08));
}

void ComplexTest::testDividing3() {
    Complex c1(0.0, 0.0);
    Complex c2(3.0, 4.0);
    Complex result = c1 / c2;
    checkTest("testDividing3", result == Complex(0.0, 0.0));
}

void ComplexTest::testEquality1() {
    Complex c1(1.0, 2.0);
    Complex c2(1.0, 2.0);
    bool result = (c1 == c2);
    checkTest("testEquality1", result);
}

void ComplexTest::testEquality2() {
    Complex c1(1.0, 2.0);
    Complex c2(3.0, 4.0);
    bool result = (c1 == c2);
    checkTest("testEquality2", !result);
}

void ComplexTest::testEquality3() {
    Complex c1(1.0, 2.0);
    Complex c2(1.0, -2.0);
    bool result = (c1 == c2);
    checkTest("testEquality3", !result);
}

void ComplexTest::testModulus1() {
    Complex c(3.0, 4.0);
    double result = c.modulus();
    checkTest("testModulus1", result == 5.0);
}

void ComplexTest::testModulus2() {
    Complex c(1.0, 1.0);
    double result = c.modulus();
    checkTest("testModulus2", abs(result - 1.41421) < 0.00001);
}

void ComplexTest::testModulus3() {
    Complex c(0.0, 0.0);
    double result = c.modulus();
    checkTest("testModulus3", result == 0.0);
}
