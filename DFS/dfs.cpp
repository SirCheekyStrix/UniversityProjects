#include<iostream>
#include<sstream>
using namespace std;

void dfs(int x, int** tab, bool* visit, int n) {
    visit[x] = true;
    for ( int i = 0; i < n; i++ ) {
        if ( tab[x][i] && !visit[i] ) {
            dfs(i, tab, visit, n);
        }
    }
}

int skladowe(int n, int** tab) {
    bool* visit = new bool[n]();
    int counter = 0;
    for ( int i = 0; i < n; i++ ) {
        if ( !visit[i] ) {
            dfs(i, tab, visit, n);
            counter += 1;
        }
    }
    delete[] visit;
    return counter;
}

int main() {
    int n;
    cin >> n;
    cin.ignore();

    int** adj = new int*[n];
    for (int i = 0; i < n; i++) {
        adj[i] = new int[n]();
    }

    for (int i = 0; i < n; i++) {
        string line;
        getline(cin, line);
        stringstream ss(line);
        int v;
        while (ss >> v) {
            adj[i][v - 1] = 1;
        }
    }

    int s = skladowe(n, adj);
    cout << s << endl;

    for (int i = 0; i < n; i++) {
        delete[] adj[i];
    }
    delete[] adj;

    return 0;
}