#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int N = 5592224;
//const int N = 1000;
const int n = 1746429, m = 150341, k = 1404, d = 20;
VectorXi user_indices(N);
VectorXi image_indices(N);

vector<int> user_indices2(N);
vector<int> image_indices2(N);

VectorXf ratings(N);

vector<int> user_start(n); 
vector<int> user_end(n);

void prepro(){
  for(int i=0;i<N;i++){
    user_end[user_indices(i)] = i+1;
  }
  for(int i=N-1;i>=0;i--){
    user_start[user_indices(i)] = i;
  }
}

void prepro2(){
  for(int i=0;i<N;i++){
    user_end[user_indices2[i]] = i+1;
  }
  for(int i=N-1;i>=0;i--){
    user_start[user_indices2[i]] = i;
  }
}

//const int n = 1000, m = 1000, k = 1404, d = 20;
MatrixXf I(m, k);

float alpha, beta, learning_rate;
int als_max_epochs, adam_max_epochs;

MatrixXf X = MatrixXf::Random(n,d);
MatrixXf Y = MatrixXf::Random(k,d);

void update_X(){
  for(int u=0; u<n; u++){

    if(u%10 == 0) cout << "Processed " << u << "\n";
  
    MatrixXf A = (alpha * (user_end[u] - user_start[u])/d ) * MatrixXf::Identity(d,d);

    VectorXf b = VectorXf::Zero(d);


    for(int i=user_start[u]; i<user_end[u]; i++){
      MatrixXf iY = I.row(image_indices2[i]) * Y;
      A += iY.transpose() * iY;
      b += ratings(i) * iY.transpose();
    }

    A.colPivHouseholderQr().solve(b);



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
    user_indices(i) = stoi(s);
    user_indices2[i] = stoi(s);


    ss >> s;
    image_indices(i) = stoi(s);
    image_indices2[i] = stoi(s);

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

  cout << X(0,0) << " " << Y(0,0) << "\n";

  //cout << user_indices({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}) << "\n\n\n";

  cout << "fuck u " << user_indices(N-1) << "\n";

  cout << "start,end" << user_start[0] << " " << user_end[0] << "\n";

  update_X();

  cout << "Fnshed\n";

  return 0;
}