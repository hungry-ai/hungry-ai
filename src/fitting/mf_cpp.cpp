#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <random>
#include <unordered_set>
#include <set>
#include <cmath> 


using namespace std;
using namespace Eigen;

const int N = 15640156;
const int n = 324749, m = 31587, k = 2636; //receiving a k+1 
int d; 

vector<int> original_user_indices;
vector<int> original_image_indices;
vector<int> original_ratings;
vector<vector<float>> original_I;

MatrixXf X;
MatrixXf Y;
MatrixXf IY;
MatrixXf I; 

string version;

//The first three have the form (user_index, image_index, rating)
vector<int> user_indices;  //Increasing integers from [0,users) 
vector<int> image_indices; 
vector<float> ratings;
vector<int> user_start; //Vector of size |users| where index = user_start[i] is first index such that user_indices[index] = i 
vector<int> user_end;

vector<int> val_user_indices;
vector<int> val_image_indices;
vector<float> val_ratings;


//float alpha, beta, learning_rate; 
float alpha = .01, beta = .01, learning_rate = 0.007;
float time_spent;

class mf_model{
  private: 
    MatrixXf Y;
    MatrixXf I;
    unordered_map<int,MatrixXf> XTX;
    unordered_map<int,VectorXf> XTy;
    unordered_map<int,VectorXf> user_weights;
    unordered_map<int,VectorXf> image_weights;
    unordered_set<int> users;
    int m;
    int k;
    int d;

  public: 
    mf_model(
      MatrixXf Y,
      MatrixXf I,
      int m,
      int k,
      int d
    ): 
      Y(Y),
      I(I), 
      m(m), 
      k(k),
      d(d) {

        MatrixXf IY = I*Y;

        for(int i=0;i<m;i++){
          image_weights[i] = IY.row(i);
        }

    }

    void add_user(int user_index){
      if(users.count(user_index)) return;

      XTX[user_index] = MatrixXf::Zero(d,d);
      XTy[user_index] = VectorXf::Zero(d);
      user_weights[user_index] = VectorXf::Zero(d);
      users.insert(user_index);
    }

    void add_review(int user_index, int image_index, float rating){
      
      if(!users.count(user_index)){
        add_user(user_index);
      }

      XTX[user_index] += alpha/d * MatrixXf::Identity(d,d) + image_weights[image_index] * (image_weights[image_index].transpose());
      XTy[user_index] += rating * image_weights[image_index];
      user_weights[user_index] = XTX[user_index].colPivHouseholderQr().solve(XTy[user_index]);

    }

    float predict(int user_index, int image_index){
      if(users.count(user_index)){
        return user_weights[user_index].dot(image_weights[image_index]);
      }
      return 3;
    }

};

bool als_converged(float new_loss, float old_loss){
  return abs(new_loss-old_loss) < 0.001;
}

bool Y_converged(float new_loss, float old_loss){
  return abs(new_loss-old_loss) < 0.01;
}

void update_X(int users){
  for(int u = 0; u < users; u++){
    auto begin = std::chrono::high_resolution_clock::now();  

    if(user_end[u] == user_start[u]) continue;

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

float loss(int reviews, int users){
  
  float loss = 0;
  float penalty_x = 0;
  float penalty_y = 0;

  IY = I * Y;

  for (int i = 0; i < reviews; ++i) {
    float sq_loss = ratings[i]-IY.row(image_indices[i]).dot(X.row(user_indices[i]));
    loss += sq_loss * sq_loss / (user_end[user_indices[i]]-user_start[user_indices[i]]);
  }

  for(int u=0; u<users; u++){
    penalty_x += (X.row(u)).squaredNorm();
  }

  for(int t=0;t<k;t++){
    penalty_y += (Y.row(t)).squaredNorm();
  }

  return loss / users + alpha * penalty_x / (users * d)+ beta * penalty_y / (k * d);
}

void update_Y_adam(int adam_max_epochs, int batch_size, int reviews, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8){
  float beta_1_pow = 1.0;
  float beta_2_pow = 1.0;

  MatrixXf mm = MatrixXf::Zero(k,d);
  MatrixXf v = MatrixXf::Zero(k,d);

  float old_loss = 0.;

  for(int epoch = 1; epoch <= adam_max_epochs; epoch++){
    MatrixXf gradient = MatrixXf::Zero(k, d);
    beta_1_pow *= beta_1;
    beta_2_pow *= beta_2;
    IY = I * Y;

    for(int u=0;u<batch_size;u++){
      if (u % 10000 == 0)
        cout << u << endl;

      int start_index = user_start[u];
      int end_index = user_end[u];

      VectorXf x_u = X.row(u), i_part = VectorXf::Zero(k);

      for (int i = start_index; i < end_index; ++i) {
        float coeff = ratings[i] - IY.row(image_indices[i]) * x_u;
        i_part += coeff * I.row(image_indices[i]);
      }
      gradient -= 1./(end_index - start_index) * i_part * x_u.transpose();
    }

    gradient = gradient * 2./(batch_size) + (2. * beta / (k * d)) * Y;

    for(int j=0;j<d;j++){
      gradient(k-1,j) = 0;
    }

    for(int i=0;i<k;i++){
      gradient(i,0) = 0;
    }

    mm = beta_1 * mm + (1-beta_1) * gradient;

    MatrixXf gradient2 = gradient.array().square();

    v = beta_2 * v + (1-beta_2) * gradient2;

    mm /= (1-beta_1_pow);

    v /= (1-beta_2_pow);

    MatrixXf v_corrected = (v).cwiseSqrt().array()+eps;

    MatrixXf final = (learning_rate * mm.array()) / v_corrected.array();

    Y.array() -= final.array(); 

    cout << "Computing loss" << endl;
    auto begin = std::chrono::high_resolution_clock::now();  
    float new_loss = loss(reviews, batch_size);
    cout << "Loss: " << new_loss << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "Loss computation took: " << elapsed.count() * 1e-9 << endl;

    if (epoch > 1 && Y_converged(new_loss,old_loss)){
      cout << "Converged early!" << endl;
      break;
    }
    old_loss = new_loss;
  }

}

void train_mf(int max_als_epochs, int adam_max_epoch, int users, int reviews){

  float old_loss = 0;
  for(int als_epoch=0; als_epoch < max_als_epochs; als_epoch++){

    cout << "Als_epoch " << als_epoch << endl;

    cout << "Computing X" << endl;
    auto begin = std::chrono::high_resolution_clock::now();  
    update_X(users);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "X computation took: " << elapsed.count() * 1e-9 << endl;

    cout << "Computing loss" << endl;
    begin = std::chrono::high_resolution_clock::now();  
    cout << "Loss: " << loss(reviews, users) << endl;
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "Loss computation took: " << elapsed.count() * 1e-9 << endl;

    cout << "Computing Y" << endl;
    begin = std::chrono::high_resolution_clock::now();  
    update_Y_adam(adam_max_epoch,users,reviews);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    cout << "Y computation took: " << elapsed.count() * 1e-9 << endl;


    float new_loss = loss(reviews,users);
    if(als_epoch && als_converged(new_loss,old_loss)){
      cout << "Als converged early!";
      break;
    }
    old_loss = new_loss;
  }
}

void init_Y(string file_name){
  cout << "Loading Y\n";
  auto begin = std::chrono::high_resolution_clock::now();  
  ifstream f;  
  f.open(file_name);

  int file_k, file_d;
  f >> file_k >> file_d;

  assert(file_k == k);
  assert(file_d == d);

  for(int i=0;i<k;i++){
    for(int j=0;j<d;j++){
      f >> Y(i,j);
    }
  }
  
  f.close();

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  cout << "Loaded Y. Took: " << elapsed.count() * 1e-9 << "\n";
}

vector<int> permutation(N);

void random_permutation(){
  random_device rd;
  mt19937 gen(rd());
  for (int i = 0; i < N; i++ ) {
    uniform_int_distribution<> dis(0, i);
    int j = dis(gen);
    permutation[i] = permutation[j];
    permutation[j] = i;
  }
}

void preprocess(int reviews, int users){

  user_start = vector<int>(users,-1);
  user_end = vector<int>(users,-1);

  for(int i=0;i<reviews;i++){ 
    user_end[user_indices[i]] = i+1;
  }
  for(int i=reviews-1;i>=0;i--){
    user_start[user_indices[i]] = i;
  }

  bool not_missing = true;
  for(auto u:user_start){
    not_missing &= (u != -1);
  }

  for(auto u:user_end){
    not_missing &= (u != -1);
  }

  assert (not_missing);
}

float cross_validation(string initial_Y_file, int K = 5, int max_als_epoch = 10, int adam_max_epoch = 10){ //Ignoring their indices, coding them [0,users)
  random_permutation();

  float result = 0;
  float sampleresult = 0;

  for(int l = 0; l < K; l++){
    cout << "Version: " << version << " started training " << l << endl;

    vector<int> train_indices;
    vector<int> val_indices;

    if( K != 1){
      for(int i=0;i<N;i++){
        if(permutation[i]%K != l){
          train_indices.push_back(i);
        }
        else{
          val_indices.push_back(i);
        }
      }
    }
    else{
      for(int i=0;i<N;i++){
        train_indices.push_back(i);
      }
    }

    int reviews = train_indices.size();

    unordered_map<int,int> users_hash;
    int index = 0;
    for(auto ti: train_indices){
      if(!users_hash.count(original_user_indices[ti])){
        users_hash[original_user_indices[ti]] = index;
        index++;
      }
    }
    int users = users_hash.size();

    user_indices.clear();
    image_indices.clear();
    ratings.clear();
    for(auto ti:train_indices){ //Initializing the information vectors with hashed user,image indices between [0,users) and [0,images)
      user_indices.push_back(users_hash[original_user_indices[ti]]);
      image_indices.push_back(original_image_indices[ti]);
      ratings.push_back(original_ratings[ti]);
    }

    vector<int> unique_users;
    for(auto u:users_hash){
      unique_users.push_back(u.first);
    }
    sort(unique_users.begin(),unique_users.end());

    X = MatrixXf::Random(users,d);
    Y = MatrixXf::Random(k,d);

    for(int j=0;j<d;j++){
      Y(k-1,j) = 0;
    }

    for(int i=0;i<k;i++){
      Y(i,0) = 0;
    }

    Y(k-1,0) = 1;
    
    //init_Y(initial_Y_file);
    
    I = MatrixXf::Zero(m,k);

    for(int i=0;i<m;i++){
      for(int j=0;j<k;j++){
        I(i,j) = original_I[i][j];
      }
    }

    IY = I*Y;

    preprocess(reviews, users); //Fill user_begin, user_end vectors 

    train_mf(max_als_epoch,adam_max_epoch,users,reviews); 

    mf_model model = mf_model(Y,I,m,k,d); //Change I to original_I ? 

    cout << "Version: " << version << " finished training " << l << endl;

    float RMSE = 0;

    for(auto vi: val_indices){ //Testing with the K-1 trained model 
      float prediction = model.predict(original_user_indices[vi], original_image_indices[vi]);
      RMSE += (original_ratings[vi] - prediction)* (original_ratings[vi] - prediction);
      model.add_review(original_user_indices[vi],original_image_indices[vi],original_ratings[vi]);
    }

    RMSE = sqrt(RMSE/val_indices.size());

    result += RMSE; 
    cout << "Model " << l << " had RMSE " << RMSE << "\n";
  }

  return result/K;

}

void read_files() {
  string line, s;

  cout << "reading reviews_train_cpp.csv" << endl;
  ifstream f1("/Users/alef/Documents/hungry-ai/data/real/reviews_train_cpp.csv");
  while(getline(f1,line)){
    stringstream ss(line);

    ss >> s;
    original_user_indices.push_back(stoi(s));

    ss >> s;
    original_image_indices.push_back(stoi(s));

    ss >> s;
    original_ratings.push_back(stof(s));
  }
  f1.close();

  cout << "reviews " << original_user_indices.size() << "\n";


  assert(original_user_indices.size() == original_image_indices.size());
  assert(original_user_indices.size() == original_ratings.size());

  cout << "done" << endl;

  cout << "reading I.txt" << endl;
  ifstream f2("/Users/alef/Documents/hungry-ai/data/real/I.txt");

  vector<int> row_sizes;
  while(getline(f2,line)) {    
    stringstream ss(line);

    vector<float> row;
    while(ss >> s){
      float temp = stof(s);
      if(temp < -1000){
        cout << s << "\n";
        return;
      }
      row.push_back(temp);
    }
    original_I.push_back(row);
    row_sizes.push_back(row.size());

  }
  f2.close();

  cout << "I dimensions: " << original_I.size() << " " << original_I[0].size() << "\n";

  sort(row_sizes.begin(),row_sizes.end());

  assert(row_sizes[0] == row_sizes[row_sizes.size()-1]);

  cout << "done" << endl;

}

void save_CV_results(string file_name, float score){
  ofstream f;
  f.open(file_name);
  f << score << endl;
  f.close();
}

void save_Y(string file_name, int tags, int factors){
  cout << "Saving Y\n";
  auto begin = std::chrono::high_resolution_clock::now();  
  ofstream f;
  f.open(file_name);
  f << tags << " " << factors << "\n";
  for(int i=0;i<tags;i++){
    for(int j=0;j<factors;j++){
      f << Y(i,j) << " ";
    }
    f << "\n";
  }
  f.close();
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  cout << "Saved Y. Took: " << elapsed.count() * 1e-9 << "\n";
}


int main(int argc, char* argv[])
{
  if (argc <=3) {
    cout << "usage: mf_cpp <alpha> <beta> <d> <version>";
    return 1;
  }

  alpha = stof(argv[1]);
  beta = stof(argv[2]);
  d = stoi(argv[3]);
  version = argv[4];
  cout << "using alpha=" << alpha
        << ", beta=" << beta
        << ", d=" << d
        << ", version=" << version
        << endl;

  
  cout << "Reading files\n";
  read_files();

  float rmse_score = cross_validation("Ybias1.txt",1,10,10);

  save_CV_results("CV_results_"+version, rmse_score);

  save_Y("Ybias1.txt", k, d);

  return 0;
}