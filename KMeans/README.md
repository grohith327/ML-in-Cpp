# KMeans Clustering
A C++ program that performs KMeans clustering and returns the accuracy </br>
### Ensure the path for mlpack has been set: </br>
By default mlpack is installed to `/usr/local/`, if you have installed somewhere specify that path. 
```
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
```
### Compile Program with the following command: </br>
```
clang++ -L/usr/local/lib -l mlpack -l armadillo -l boost_serialization -l boost_program_options -I/usr/local/include KMeans.cpp
```
### Run: </br>
The program takes two command line arguments </br>
* `features.csv` - input features
* `responses.csv` - predictors

```
./a.out <features.csv> <responses.csv>
```

#### Note: </br>
* KMeans clustering is an unsupervised learning algorithm, i.e it does not require any labels. However, in this example we use responses to calculate the accuracy of a model. 
* The code can be slightly altered to work without the labels and return only the cluster assignments
* There is no train/test split hapenning here. All of the data is used to identify clusters and for each point, cluster assignments are calculated. 