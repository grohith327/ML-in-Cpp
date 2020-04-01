#include <iostream>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

float accuracy(arma::mat true_labels, arma::Row<size_t> predictions)
{
    int correct = 0;
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

arma::Row<size_t> getLabels(arma::mat& predOut)
{
  arma::Row<size_t> pred(predOut.n_cols);

  for (size_t j = 0; j < predOut.n_cols; ++j)
  {
    pred(j) = arma::as_scalar(arma::find(
        arma::max(predOut.col(j)) == predOut.col(j), 1)) + 1;
  }

  return pred;
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

    const int numClasses = arma::size(arma::unique(trainY))(0);

    trainY += 1;
    testY += 1;

    cout<<"Train data size: "<<arma::size(trainX)<<", Train labels size: "<<arma::size(trainY)<<endl;
    cout<<"Test data size: "<<arma::size(testX)<<", Test labels size: "<<arma::size(testY)<<endl;
    
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    model.Add<Linear<> >(trainX.n_rows, 10);
    model.Add<ReLULayer<> >();
    model.Add<Linear<> >(10, 3);
    model.Add<LogSoftMax<> >();

    ens::StandardSGD optimizer; 

    for(size_t i = 0;i < EPOCHS;i++)
    {
        model.Train(trainX, trainY, std::move(optimizer));
        optimizer.ResetPolicy() = false;

        arma::mat train_predictions, test_predictions;
        arma::Row<size_t> train_predLabels, test_predLabels;

        model.Predict(trainX, train_predictions);
        train_predLabels = getLabels(train_predictions);
        float train_acc = accuracy(trainY.t(), train_predLabels);

        model.Predict(testX, test_predictions);
        test_predLabels = getLabels(test_predictions);
        float test_acc = accuracy(testY.t(), test_predLabels);

        cout<<"Epoch:"<<i+1<<", Train accuracy:"<<train_acc<<", Test accuracy:"<<test_acc<<endl;
    }

    return 0;
}