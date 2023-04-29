#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <ctime>
#include <chrono>

using namespace std;
using namespace Eigen;

const int N = 5592224;
const int n = 1746429, m = 150341, k = 1404, d = 20;

vector<int> user_indices(N);
vector<int> image_indices(N);
vector<int> ratings(N);
vector<int> user_start(n); 
vector<int> user_end(n);

MatrixXf I(m, k);
MatrixXf X = MatrixXf::Random(n,d);
MatrixXf Y = MatrixXf::Random(k,d);

float alpha, beta, learning_rate;
float time_spent;
int als_max_epochs, adam_max_epochs;

void update_X(){
  for(int u = 0; u < n; u++){
    auto begin = std::chrono::high_resolution_clock::now();  

    MatrixXf A = (alpha * (user_end[u] - user_start[u])/d ) * MatrixXf::Identity(d,d);
    VectorXf b = VectorXf::Zero(d);

    for(int i=user_start[u]; i<user_end[u]; i++){
      MatrixXf iY = I.row(image_indices[i]) * Y;
      A += iY.transpose() * iY;
      b += ratings[i] * iY.transpose();
    }
    
    VectorXf xu = A.colPivHouseholderQr().solve(b);
    for(int col = 0; col < d; col++){
      X(u,col) = xu(col);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    time_spent += elapsed.count() * 1e-9;
    if(u%10000 == 0 && u) cout << "Time avg " << time_spent/(u/10000) << "\n";
  }
}

void update_Y_adam(int adam_max_epochs, int batch_size, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8){
  float beta_1_pow = 1.0;
  float beta_2_pow = 1.0;

  MatrixXf mm = MatrixXf::Zero(k,d);
  MatrixXf v = MatrixXf::Zero(k,d);

  for(int epoch = 1; epoch <= adam_max_epochs; epoch++){
    MatrixXf gradient = MatrixXf::Zero(k,d);
    beta_1_pow *= beta_1;
    beta_2_pow *= beta_2;

    for(int u=0;u<batch_size;u++){
      int start_index = user_start[u];
      int end_index = user_end[u];

      VectorXf x_u = X.row(u);

      VectorXf r_u = VectorXf::Zero(end_index-start_index);
      for(int rtg = start_index; rtg < end_index; rtg++){
        r_u(rtg-start_index) = ratings[rtg];
      }
      
      vector<int> image_indices_range;
      for(int rtg = start_index; rtg < end_index; rtg++){
        image_indices_range.push_back(image_indices[rtg]);
      }
      
      MatrixXf I_u = I(image_indices_range,all);

      MatrixXf temp = ((r_u - (I_u * Y)* x_u).transpose() * I_u).transpose() * x_u.transpose();
      temp.array() *= 2.0/(end_index-start_index);
      
      gradient.array() -= temp.array();

      mm = beta_1 * mm + (1-beta_1) * gradient;

      MatrixXf gradient2 = gradient.array().square();

      v = beta_2 * v + (1-beta_2) * gradient2;

      MatrixXf v_corrected = (v / (1.0 - beta_2_pow)).cwiseSqrt();
      v_corrected.array() += eps;

      mm.array() *= learning_rate/(1-beta_1_pow);

      mm.array() /= v_corrected.array();

      Y.array() -= mm.array();

    }
  }

}

float loss(){
  
  float loss = 0;
  float penalty_x = 0;
  float penalty_y = 0;

  for(int u=0; u<n; u++){
    for(int i=user_start[u];i<user_end[u];i++){      
      float sq_loss = ratings[i]-I.row(image_indices[i]).dot(Y * X.row(u).transpose());
      loss += sq_loss * sq_loss;
    }

    penalty_x += (X.row(u)).squaredNorm();
  }

  for(int t=0;t<k;t++){
    penalty_y += (Y.row(t)).squaredNorm();
  }

  return loss / n + alpha / (n * d) * penalty_x + beta / (k * d) * penalty_y;
}

void train_mf(int max_als_epochs, int adam_max_epochs){
  for(int als_epoch=0; als_epoch < max_als_epochs; als_epoch++){

    cout << "Als_epoch " << als_epoch << "\n";

    cout << "Computing X\n";
    auto begin = std::chrono::high_resolution_clock::now();  
    update_X();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "X computation took: " << elapsed.count() * 1e-9 << "\n";

    cout << "Computing Y\n";
    begin = std::chrono::high_resolution_clock::now();  
    update_Y_adam(adam_max_epochs,n);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "Y computation took: " << elapsed.count() * 1e-9 << "\n";


    cout << "Computing loss\n"; 
    begin = std::chrono::high_resolution_clock::now();  
    cout << "Loss: " << loss() << "\n";
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "Loss computation took: " << elapsed.count() * 1e-9 << "\n";
  }
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

void pre_processing(){
  for(int i=0;i<N;i++){
    user_end[user_indices[i]] = i+1;
  }
  for(int i=N-1;i>=0;i--){
    user_start[user_indices[i]] = i;
  }
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
  pre_processing();

  train_mf(2,2);


  return 0;
}