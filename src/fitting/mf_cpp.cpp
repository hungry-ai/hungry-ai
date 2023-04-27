#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace std;

const int N = 5592224;
Eigen::VectorXi user_indices(N);
Eigen::VectorXi image_indices(N);
Eigen::VectorXf ratings(N);

const int n = 1746429, m = 150341, k = 1404, d = 20;
Eigen::MatrixXf I(m, k);

float alpha, beta, learning_rate;
int als_max_epochs, adam_max_epochs;

void read_files() {
  string line, s;

  cout << "reading reviews_train_cpp.csv" << endl;
  ifstream f1("../../data/reviews/reviews_train_cpp.csv");
  for (auto i = 0; i < n; ++i) {
    getline(f1, line);
    stringstream ss(line);

    ss >> s;
    user_indices(i) = stoi(s);

    ss >> s;
    image_indices(i) = stoi(s);

    ss >> s;
    ratings(i) = stof(s);
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
      I(i, j) = stof(s);
    }
  }
  f2.close();
  cout << "done" << endl;
}

int main(int argc, char* argv[])
{
  if (argc <= 5) {
    cout << "usage: mf_cpp <alpha> <beta> <learning_rate> <als_max_epochs> <adam_max_epochs>";
    return 1;
  }
  alpha = stof(argv[1]);
  beta = stof(argv[2]);
  learning_rate = stof(argv[3]);
  als_max_epochs = stoi(argv[4]);
  adam_max_epochs = stoi(argv[5]);
  cout << "using alpha=" << alpha
       << ", beta=" << beta
       << ", learning_rate=" << learning_rate
       << ", als_max_epochs=" << als_max_epochs
       << ", adam_max_epochs=" << adam_max_epochs
       << endl;

  read_files();
}