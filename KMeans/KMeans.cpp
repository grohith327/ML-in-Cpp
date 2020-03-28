#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::kmeans;

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
    arma::Row<size_t> responses;

    data::Load(argv[1], data, true, false);
    data::Load(argv[2], responses, true);

    const size_t clusters = arma::size(arma::unique(responses))(1);

    cout<<"Data shape: "<<arma::size(data)<<" Clusters: "<<clusters<<endl;

    arma::Row<size_t> cluster_assign;

    data = data.t();

    KMeans<> model;
    model.Cluster(data, clusters, cluster_assign);

    float acc = accuracy(responses, cluster_assign);
    cout<<"Accuracy: "<<acc<<endl;

    return 0;
}