#include <iostream>
#include <math.h>
#include <assert.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace arma;

void split_data(mat data, arma::Row<size_t> responses, int train_ex, 
                mat& train_data, mat& test_data, arma::Row<size_t>& train_resp,
                arma::Row<size_t>& test_resp)
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

float accuracy(arma::Row<size_t> true_labels, arma::Row<size_t> predictions)
{
    int correct = 0;
    assert(arma::size(true_labels) == arma::size(predictions));
    int len = true_labels.size();
    for(int i = 0;i < len;i++)
    {
        if(true_labels(i) == predictions(i))
        {
            correct++;
        }
    }

    float acc = float(correct) / float(len);
    return acc*100;
}

int main(int argc, char** argv)
{
    arma::Row<size_t> temp(10, arma::fill::zeros);

    arma::mat data;
    arma::Row<size_t> responses;
    
    data::Load(argv[1], data, true, false);
    data::Load(argv[2], responses, true);

    float train_size = 0.8;
    int train_ex = int(train_size * arma::size(data)[0]);
    int test_ex = arma::size(data)[0] - train_ex;

    arma::mat train_data(train_ex, arma::size(data)[1], arma::fill::zeros), test_data(test_ex, arma::size(data)[1], arma::fill::zeros);
    arma::Row<size_t> train_resp(train_ex, arma::fill::zeros), test_resp(test_ex, arma::fill::zeros);
    
    split_data(data, responses, train_ex, train_data, test_data, train_resp, test_resp);
    cout<<"Training data shape "<<arma::size(train_data)<<", Training responses size "<<train_resp.size()<<endl;
    cout<<"Testing data shape "<<arma::size(test_data)<<", Testing responses size "<<test_resp.size()<<endl;

    train_data = train_data.t();
    test_data = test_data.t();

    arma::Row<size_t> predictions;
    
    LogisticRegression<arma::mat> model(train_data, train_resp);
    model.Classify(test_data, predictions);
    
    float acc = accuracy(test_resp, predictions);
    cout<<acc<<endl;
    return 0;
}
