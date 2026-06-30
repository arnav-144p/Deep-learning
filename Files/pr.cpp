#include <bits/stdc++.h>
using namespace std;

string intRoman(int n) {
   unordered_map <char, int> map = {
    {'I', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000} 
   };

   cout<<map.at('L');
}

int main() {
   cout<<intRoman(3749);
}