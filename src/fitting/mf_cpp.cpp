#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const int N = 5592224;
vector<int> user_indices(N, 0);
vector<int> image_indices(N, 0);
vector<float> ratings(N, 5.);

const int n = 1746429, m = 150341, k = 1404, d = 20;
float I[m][k];

void read_files() {
  string line, s;

  cout << "reading reviews_train_cpp.csv" << endl;
  ifstream f1("../../data/reviews/reviews_train_cpp.csv");
  for (auto i = 0; i < n; ++i) {
    getline(f1, line);
    stringstream ss(line);

    ss >> s;
    user_indices[i] = stoi(s);

    ss >> s;
    image_indices[i] = stoi(s);

    ss >> s;
    ratings[i] = stof(s);
  }
  f1.close();
  cout << "done" << endl;

  cout << "reading I.txt" << endl;
  ifstream f2("../../data/reviews/I.txt");
  for (auto i = 0; i < m; ++i) {
    if (i % 10000 == 0)
      cout << "processing line: " << i << endl;
    
    getline(f2, line);
    stringstream ss(line);

    for (auto j = 0; j < k; ++j) {
      ss >> s;
      I[i][j] = stof(s);
    }
  }
  f2.close();
  cout << "done" << endl;
}

int main()
{
  read_files();
}