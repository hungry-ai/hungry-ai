#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include<ctime>

using namespace std;
using namespace Eigen;

const int N = 5592224;
//const int N = 1000;
const int n = 1746429, m = 150341, k = 1404, d = 20;
//const int n = 1000, m = 1000, k = 1000, d = 20;

vector<int> user_indices(N);
vector<int> image_indices(N);
vector<int> ratings(N);
vector<int> user_start(n); 
vector<int> user_end(n);

vector<vector<float>> MM(vector<vector<float>>& A, vector<vector<float>>& B ){
    int n = A.size();
    int m = A[0].size();
    int k = B[0].size();

    vector<vector<float>> R(n,vector<float>(k,0));

    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            for(int l=0;l<m;l++){
                R[i][j] += A[i][l]*B[l][j];
            }
        }
    }

    return R;
}

void prepro(){
  for(int i=0;i<N;i++){
    user_end[user_indices[i]] = i+1;
  }
  for(int i=N-1;i>=0;i--){
    user_start[user_indices[i]] = i;
  }
}

MatrixXf I(m, k);

float alpha, beta, learning_rate;
int als_max_epochs, adam_max_epochs;

MatrixXf X = MatrixXf::Random(n,d);
MatrixXf Y = MatrixXf::Random(k,d);

float time_construct;
float time_inverse;

void update_X(){
  for(int u=0; u<n; u++){ //n

    auto time1 = time(NULL);

    if(u%10000 == 0) cout << "Processed " << u << "\n";
  
    MatrixXf A = (alpha * (user_end[u] - user_start[u])/d ) * MatrixXf::Identity(d,d);

    VectorXf b = VectorXf::Zero(d);

    for(int i=user_start[u]; i<user_end[u]; i++){
      MatrixXf iY = I.row(image_indices[i]) * Y;
      A += iY.transpose() * iY;
      b += ratings[i] * iY.transpose();
    }

    auto time2 = time(NULL);

    A.colPivHouseholderQr().solve(b);

    auto time3 = time(NULL);

    time_construct += time2-time1;
    time_inverse += time3-time2;

    if(u%10000 == 0 && u) cout << "Time avg " << time_construct/(u/10000) << "\n";


    //cout << "Finished " << u << "\n";

  }
}

void update_Y(){
  return;
}

float loss(){
  return 0;
}



void read_files() {
  string line, s;

  cout << "reading reviews_train_cpp.csv" << endl;
  ifstream f1("../../data/reviews/reviews_train_cpp.csv");
  for (auto i = 0; i < N; ++i) {
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
      I(i, j) = stof(s);
    }
  }
  f2.close();
  cout << "done" << endl;
}

int main(int argc, char* argv[])
{
 /*
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

*/

  read_files();
  prepro();
  update_X();

  cout << "Time constructing " << time_construct << "\n";
  cout << "Time inverting " << time_inverse << "\n";

  return 0;
}