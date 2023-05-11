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


class mf_model{
  private: 
    MatrixXf X;
    MatrixXf Y;
    MatrixXf I;
    VectorXf Xavg;
    VectorXf Iavg;
    vector<int> user_indices;
    vector<int> image_indices;
    unordered_map<int,int> user_hash;
    unordered_map<int,int> image_hash;
    int n;
    int m;
    int k; 

  public: 
    mf_model(
      MatrixXf X, 
      MatrixXf Y,
      MatrixXf I, 
      vector<int> user_indices, 
      vector<int> image_indices, 
      unordered_map<int,int> user_hash,
      unordered_map<int,int> image_hash,
      int n,
      int m,
      int k
    ): 
      X(X), 
      Y(Y),
      I(I), 
      user_indices(user_indices), 
      image_indices(image_indices), 
      user_hash(user_hash),
      image_hash(image_hash),
      n(n), 
      m(m), 
      k(k) {
        
        Xavg = X.colwise().mean();
        Iavg = I.colwise().mean();
    }

    float predict(int user_id, int image_id){
    
    VectorXf xu = Xavg;
    if(user_hash.count(user_id)) xu = X.row(user_hash[user_id]);

    VectorXf ii =  Iavg;
    if(image_hash.count(image_id)) ii =  I.row(image_hash[image_id]);

    VectorXf xuY= Y * xu;
    
    return ii.dot(xuY);
  }
};


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

    if(u == 2236){
      cout << u << " " << A(0,0) << " " << (A(0,0) < -1000) << "\n";
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

    if (epoch > 1 && abs(new_loss - old_loss) < .01) {
      cout << "Converged early!" << endl;
      break;
    }
    old_loss = new_loss;
  }

}

void train_mf(int max_als_epochs, int adam_max_epoch, int users, int reviews){
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
  }
}

void init_Y(string file){
  cout << "Loading Y\n";
  auto begin = std::chrono::high_resolution_clock::now();  
  ifstream f;  
  f.open(file);

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

void preprocess(int reviews, int users){ //Ranges [user_start,user_end) for each index [0,users)

  user_start = vector<int>(users,-1);
  user_end = vector<int>(users,-1);

  for(int i=0;i<reviews;i++){ //Traverse user_indices to find last index. 
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

pair<float,float> cross_validation(string initial_Y_file, int K = 5, int max_als_epoch = 10, int adam_max_epoch = 10){ //Ignoring their indices, coding them [0,users) [0,images] use the has for the vectors.
  random_permutation();

  float result = 0;
  float sampleresult = 0;
  //cout << "Started K split for cross-validation.\n";

  for(int l = 0; l < K; l++){
    cout << "Version: " << version << " started training " << l << endl;

    vector<int> train_indices;
    vector<int> val_indices;

    if( K != 1){
      for(int i=0;i<N;i++){
        if(permutation[i]%K != l){ //RELATE TO PERMUTATION
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

    set<int> images_set;
    for(auto ti: train_indices){
      images_set.insert(original_image_indices[ti]);
    }

    unordered_map<int,int> images_hash;
    index = 0;
    for(auto im: images_set){
      if(!images_hash.count(im)){
        images_hash[im] = index;
        index++;
      }
    }
    int images = images_hash.size();

    user_indices.clear();
    image_indices.clear();
    ratings.clear();
    for(auto ti:train_indices){ //Initializing the information vectors with hashed user,image indices between [0,users) and [0,images)
      user_indices.push_back(users_hash[original_user_indices[ti]]);
      image_indices.push_back(images_hash[original_image_indices[ti]]);
      ratings.push_back(original_ratings[ti]);
    }

    vector<int> unique_users;
    for(auto u:users_hash){
      unique_users.push_back(u.first);
    }
    sort(unique_users.begin(),unique_users.end());

    vector<int> unique_images;
    for(auto u:images_hash){
      unique_images.push_back(u.first);
    }
    sort(unique_images.begin(),unique_images.end());

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

    for(int i=0;i<images;i++){
      for(int j=0;j<k;j++){
        I(i,j) = original_I[unique_images[i]][j];
      }
    }


    IY = I*Y;

    preprocess(reviews, users); //Fill user_begin, user_end vectors 


    /* Sanity checking construction 
    int Q = 10;
    cout << "First few train indices " ;
    for(int i=0;i<Q;i++){
      cout << train_indices[i] << " ";
    }
    cout << "\n";

    cout << "Users : ";
    for(int i=0;i<Q;i++){
      cout << original_user_indices[i] << " ";
    }
    cout << "\n";

    cout << "Images: ";
    for(int i=0;i<Q;i++){
      cout << original_image_indices[i] << " ";
    }
    cout << "\n";

    cout << "Our users : ";
    for(int i=0;i<Q;i++){
      cout << user_indices[i] << " ";
    }
    cout << "\n";

    cout << "Our images: ";
    for(int i=0;i<Q;i++){
      cout << image_indices[i] << " ";
    }
    cout << "\n";

    return; */

    train_mf(max_als_epoch,adam_max_epoch,users,reviews); 

    mf_model model = mf_model(X,Y,I,user_indices,image_indices,users_hash,images_hash,users,images,k);

    cout << "Version: " << version << " finished training " << l << endl;

    float RMSE = 0;

    for(auto vi: val_indices){ //Testing with the K-1 trained model 
      float prediction = model.predict(original_user_indices[vi], original_image_indices[vi]);
      RMSE += (original_ratings[vi] - prediction)* (original_ratings[vi] - prediction);

      //if(prediction > 8 || prediction < -3) cout << "bad pred " << prediction << "\n"; 

    }

    float sampleRMSE = 0;
    for(int i=0;i<train_indices.size();i++){
      float prediction = model.predict(original_user_indices[train_indices[i]],original_image_indices[train_indices[i]]);
      sampleRMSE += (ratings[i] - prediction)*(ratings[i] - prediction);

      if(i < 10){
        //cout << "Prediction " << prediction << " rating " << ratings[i] << "\n";
      }

    }

    RMSE = sqrt(RMSE/val_indices.size());
    sampleRMSE = sqrt(sampleRMSE/train_indices.size());

    result += RMSE; 
    sampleresult += sampleRMSE;
    cout << "Model " << l << " had RMSE " << RMSE << "\n";
    cout << "In sample RMSE " << sampleRMSE << "\n";
    //evaluate on remaining slit 
  }

  return {result/K, sampleresult/K};

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


void save_Y(string version, int tags, int factors){ //First line should have dimensions 
  cout << "Saving Y\n";
  auto begin = std::chrono::high_resolution_clock::now();  
  ofstream f;
  f.open("Y"+version);
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

  for(auto v:original_I){
    for(auto u:v){
      if(u < -1000){
        cout << u << "\n";
      }
    }
  }

  //preprocess();

  //init_XY("1.txt");

  //train_mf(1,1);

  /* Sanity check for soft start 
  
  X = MatrixXf::Random(n,d);
  Y = MatrixXf::Random(k,d);
  init_Y("Y1.txt");
  cout << "1\n";
  I = original_I;
  cout << "2\n";
  IY = I*Y;
  cout << "3\n";

  image_indices = original_image_indices;
  user_indices = original_user_indices;
  ratings = original_ratings;

  cout << "Soft thing loss is " << loss(N,n) << "\n";

  cout << "Pre-preprocess" << "\n";
  preprocess(N,n);

  cout << "Preprocessed\n";
  update_X(n);

  cout << "Soft thing loss is " << loss(N,n) << "\n"; */

  /*float a1 = cross_validation(5,0.001,0.001,0.0001);
  float a2 = cross_validation(5,0.001,0.001,0.001);
  float a3 = cross_validation(5,0.001,0.001,0.01);
  float a4 = cross_validation(5,0.001,0.001,0.1);

  cout << "CV average 0.0001: " << a1 << "\n";

  cout << "CV average 0.001: " << a2 << "\n";

  cout << "CV average 0.01: " << a3 << "\n";

  cout << "CV average 0.1: " << a4 << "\n"; */

  //cout << "CV average: " << cross_validation(5,0.001,0.001,0.05) << "\n";


  cout << "About to start" << endl;
  pair<float,float> scores = cross_validation("Yreal6.txt", 2,1,1); //10,10 last two 

  ofstream f;
  f.open("CV"+version);
  f << scores.first << endl;
  f << scores.second << endl;
  f.close();


  save_Y("bias1.txt", k, d);
  
  //save_Y("18.txt");

  //save_XY("2.txt");

  return 0;
}