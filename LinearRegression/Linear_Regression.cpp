#include <iostream>
#include <math.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;

void split_data(mat data, rowvec responses, int train_ex, 
                mat& train_data, mat& test_data, rowvec& train_resp,
                rowvec& test_resp)
{
    for(int i = 0; i < train_ex; i++)
    {
        train_data.row(i) = data.row(i);
        train_resp(i) = responses(i);
    }

    for(int i = train_ex; i < responses.size(); i++)
    {
        test_data.row(i - train_ex) = data.row(i);
        test_resp(i - train_ex) = responses(i);
    }

}

float MSE(rowvec true_resp, rowvec predictions)
{
    float total = 0.0;
    for(int i=0;i<predictions.size();i++)
    {
        total += pow(predictions(i) - true_resp(i), 2);
    }
    float mse = total / true_resp.size();
    return mse;
}

int main(int argc,  char **argv)
{
    mat data;
    rowvec responses;

    data::Load(argv[1], data, true, false);
    data::Load(argv[2], responses, true);

    float train_size = 0.8;
    int train_ex = int(train_size * arma::size(data)[0]);
    int test_ex = arma::size(data)[0] - train_ex;

    mat train_data(train_ex, arma::size(data)[1], arma::fill::zeros), test_data(test_ex, arma::size(data)[1], arma::fill::zeros);
    rowvec train_resp(train_ex, arma::fill::zeros), test_resp(test_ex, arma::fill::zeros);
    
    split_data(data, responses, train_ex, train_data, test_data, train_resp, test_resp);
    cout<<"Training data shape "<<arma::size(train_data)<<", Training responses size "<<train_resp.size()<<endl;
    cout<<"Testing data shape "<<arma::size(test_data)<<", Testing responses size "<<test_resp.size()<<endl;

    rowvec predictions;

    if(argv[3] == NULL)
    {
        LinearRegression model(train_data, train_resp);
        model.Predict(test_data, predictions);

    }
    else
    {
        double lambda = *argv[3];
        LinearRegression model(train_data, train_resp, lambda);
        model.Predict(test_data, predictions);
        throw predictions;
    }
    
    cout<<"MSE Error: "<<MSE(test_resp, predictions)<<endl;

    return 0;
}