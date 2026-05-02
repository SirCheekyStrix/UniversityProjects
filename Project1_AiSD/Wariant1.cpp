#include<iostream>
#include<string>

int max(int x, int  y ) {
    return (x > y) ? x : y;
}

std::string lcs( const std::string& str_1, const std::string& str_2 ) {
    int a = str_1.size();
    int b = str_2.size();

    int** tab = new int*[a +1];
    for ( int i = 0; i <= a; ++i ) {
        tab[i] = new int[b + 1];
    }
    for ( int i = 0; i <= a; ++i ) {
        for ( int j = 0; j <= b; ++j ) {
            if ( i == 0 || j == 0 ) {
                tab[i][j] = 0;
            }
            else if  ( str_1[i - 1] == str_1[j - 1] ) { //gdy takie same litery
                tab[i][j] = tab[i - 1][j - 1] + 1;
            }
            else {
                tab[i][j] = max( tab[i - 1][j], tab[i][j - 1] );
            }
        }
    }
    int i = a, j = b;
    std::string lcs;
    while ( i > 0 && j > 0 ) {
        if ( str_1[i - 1] == str_2[j - 1] ) {
            lcs = str_1[i - 1] + lcs;
            i -= 1;
            j -= 1;
        }
        else if ( tab[i - 1][j] > tab[i][j - 1] ) {
            i -= 1;
        }
        else {
            j -= 1;
        }
    }
    for ( int i = 0; i <= a; ++i ) {
        delete[] tab[i];
    }
    delete[] tab;
    
    return lcs;
}

int main() {
    std::string a, b;
    std::cin >> a;
    std::cin >> b;

    for ( char c : a ) {
        if ( c != 'A' && c != 'C' && c != 'G' && c != 'T' ) {
            std::cout << "Niedozwolone litery w ciągu DNA!!!"<< std::endl;
            return 1;
        }
    }
    for ( char c : b ) {
        if ( c != 'A' && c != 'C' && c != 'G' && c != 'T' ) {
            std::cout << "Niedozwolone litery w ciągu DNA!!!"<< std::endl;
            return 2;
        }
    }

    std::string _lcs = lcs(a, b);
    std::cout << _lcs<< std::endl;
    std::cout << _lcs.size() << std::endl;
    return 0;
}