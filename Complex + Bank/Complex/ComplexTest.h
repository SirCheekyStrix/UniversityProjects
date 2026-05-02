#ifndef COMPLEX_TEST_H
#define COMPLEX_TEST_H

#include "Complex.h"

class ComplexTest {
public:
    void testAll();
    void runTest(int testNumber);

    void testAdding1();
    void testAdding2();
    void testAdding3();

    void testSubtracting1();
    void testSubtracting2();
    void testSubtracting3();

    void testMultiplying1();
    void testMultiplying2();
    void testMultiplying3();

    void testDividing1();
    void testDividing2();
    void testDividing3();

    void testEquality1();
    void testEquality2();
    void testEquality3();

    void testModulus1();
    void testModulus2();
    void testModulus3();

private:
    int test_passed = 0;
    int test_failed = 0;

    void checkTest(const std::string& testName, bool condition);
};

#endif