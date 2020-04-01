#include <iostream>
#include <math.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

double compute_mse(arma::mat true_value, arma::mat predictions)
{
    return arma::accu(arma::pow(true_value - predictions, 2))/(double)true_value.n_cols;
}

int main(int argc, char** argv)
{
    arma::mat data;
    size_t EPOCHS = stoi(argv[2]);

    data::Load(argv[1], data, true);

    arma::mat train_data, test_data;
    data::Split(data, train_data, test_data, 0.2);

    arma::mat trainX = train_data.submat(1, 1, train_data.n_rows - 2, train_data.n_cols - 1);
    arma::mat trainY = train_data.submat(train_data.n_rows - 1, 1, train_data.n_rows - 1, train_data.n_cols - 1);

    arma::mat testX = test_data.submat(1, 0, test_data.n_rows - 2, test_data.n_cols - 1);
    arma::mat testY = test_data.row(test_data.n_rows - 1);

    cout<<"Train data size: "<<arma::size(trainX)<<", Train labels size: "<<arma::size(trainY)<<endl;
    cout<<"Test data size: "<<arma::size(testX)<<", Test labels size: "<<arma::size(testY)<<endl;

    FFN<MeanSquaredError<>, RandomInitialization> model;
    model.Add<Linear<> >(trainX.n_rows, 10);
    model.Add<ReLULayer<> >();
    model.Add<Linear<> >(10, 5);
    model.Add<ReLULayer<> >();
    model.Add<Linear<> >(5, 1);

    ens::StandardSGD optimizer;

    for(size_t i = 0;i < EPOCHS;i++)
    {
        model.Train(trainX, trainY, std::move(optimizer));
        optimizer.ResetPolicy() = false;

        arma::mat train_predictions, test_predictions;

        model.Predict(trainX, train_predictions);

        double train_mse = compute_mse(trainY, train_predictions);
        
        model.Predict(testX, test_predictions);

        double test_mse = compute_mse(testY, test_predictions);

        cout<<"Epoch:"<<i+1<<", Train MSE:"<<train_mse<<", Test MSE:"<<test_mse<<endl;
    }

    return 0;
}