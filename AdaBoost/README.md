# AdaBoost
A C++ program that performs a Adaptive Boosting and returns the accuracy of the model </br>
### Ensure the path for mlpack has been set: </br>
By default mlpack is installed to `/usr/local/`, if you have installed somewhere specify that path. 
```
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
```
### Compile Program with the following command: </br>
```
clang++ -L/usr/local/lib -l mlpack -l armadillo -l boost_serialization -l boost_program_options -I/usr/local/include AdaBoost.cpp
```
### Run: </br>
The program takes three command line arguments </br>
* `features.csv` - input features
* `responses.csv` - predictors
* `WeakLearnerType` - Type of weak learner to use, ex: perceptron or decision stump

```
./a.out <features.csv> <responses.csv> perceptron
```

#### Note: </br>
The given data is split into training and testing with a ratio of 80/20. Accuracy is reported from test data.