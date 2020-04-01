#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;

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
    arma::mat data;

    data::Load(argv[1], data, true);

    arma::mat train_data, test_data;
    data::Split(data, train_data, test_data, 0.2);

    arma::mat trainX = train_data.submat(1, 1, train_data.n_rows - 2, train_data.n_cols - 1);
    arma::Row<size_t> trainY(train_data.n_cols - 1, arma::fill::zeros);
    for(int i = 1;i<train_data.n_cols;i++)
    {
        trainY(i - 1) = train_data.row(train_data.n_rows - 1)(i);
    }

    arma::mat testX = test_data.submat(1, 0, test_data.n_rows - 2, test_data.n_cols - 1);
    arma::Row<size_t> testY(test_data.n_cols, arma::fill::zeros);
    for(int i = 0;i<test_data.n_cols;i++)
    {
        testY(i) = test_data.row(test_data.n_rows - 1)(i);
    }

    const size_t numClasses = arma::size(arma::unique(trainY))(1);

    cout<<"Train data size: "<<arma::size(trainX)<<", Train labels size: "<<arma::size(trainY)<<endl;
    cout<<"Test data size: "<<arma::size(testX)<<", Test labels size: "<<arma::size(testY)<<endl;

    DecisionTree<> model(trainX, trainY, numClasses);

    arma::Row<size_t> predictions;
    model.Classify(testX, predictions);
    
    float acc = accuracy(testY, predictions);
    cout<<"Accuracy: "<<acc<<endl;

    return 0;
}