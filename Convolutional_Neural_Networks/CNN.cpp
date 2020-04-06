#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <ensmallen.hpp>

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
    const int EPOCHS = stoi(argv[2]);
    const int BATCH_SIZE = 32;

    data::Load(argv[1], data, true);
    data = data.submat(0, 0, data.n_rows - 1, 100);

    arma::mat train_data, test_data;
    data::Split(data, train_data, test_data, 0.2);

    arma::mat trainX = train_data.submat(1, 1, train_data.n_rows - 1, train_data.n_cols - 1 ) / 255.0;
    arma::mat trainY = train_data.submat(train_data.n_rows - 1, 1, train_data.n_rows - 1, train_data.n_cols - 1);

    arma::mat testX = test_data.submat(1, 0, test_data.n_rows - 1, test_data.n_cols - 1) / 255.0;
    arma::mat testY = test_data.row(test_data.n_rows - 1);

    trainY += 1;
    testY += 1;

    cout<<"Train data size: "<<arma::size(trainX)<<", Train labels size: "<<arma::size(trainY)<<endl;
    cout<<"Test data size: "<<arma::size(testX)<<", Test labels size: "<<arma::size(testY)<<endl;

    int width = stoi(argv[3]);
    int height = stoi(argv[4]);
    
    FFN<> model;
    model.Add<Convolution<> >(1, 16, 5, 5, 1, 1, 2, 2, width, height);
    model.Add<ReLULayer<> >();
    model.Add<MaxPooling<> >(2, 2, 2, 2, true);
    model.Add<Convolution<> >(16, 32, 5, 5, 1, 1, 2, 2, width / 2, height / 2);
    model.Add<ReLULayer<> >();
    model.Add<MaxPooling<> >(2, 2, 2, 2, true);
    model.Add<Convolution<> >(32, 64, 5, 5, 1, 1, 2, 2, width / 4, height / 4);
    model.Add<ReLULayer<> >();
    model.Add<Linear<> >(7*7*64, 10);
    model.Add<LogSoftMax<> >();

    ens::StandardSGD optimizer(5e-5, BATCH_SIZE, train_data.n_cols*EPOCHS);

    model.Train(trainX, trainY, optimizer, ens::PrintLoss());

    arma::mat train_predictions, test_predictions;
    arma::Row<size_t> train_predLabels, test_predLabels;
    
    model.Predict(trainX, train_predictions);
    train_predLabels = getLabels(train_predictions);
    float train_acc = accuracy(trainY.t(), train_predLabels);

    model.Predict(testX, test_predictions);
    test_predLabels = getLabels(test_predictions);
    float test_acc = accuracy(testY.t(), test_predLabels);

    cout<<"Train accuracy:"<<train_acc<<", Test accuracy:"<<test_acc<<endl;

    return 0;
}