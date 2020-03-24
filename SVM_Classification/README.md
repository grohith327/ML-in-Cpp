# Support Vector Machine
A C++ program that performs classification using a Support Vector Machine algorithm and returns the accuracy of the model</br>
### Ensure the path for mlpack has been set: </br>
By default mlpack is installed to `/usr/local/`, if you have installed somewhere specify that path. 
```
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
```
### Compile Program with the following command: </br>
```
clang++ -L/usr/local/lib -l mlpack -l armadillo -l boost_serialization -l boost_program_options -I/usr/local/include SVM.cpp
```
### Run: </br>
The program takes two command line arguments </br>
* `features.csv` - input features
* `responses.csv` - predictors

```
./a.out <features.csv> <responses.csv>
```

#### Note: </br>
* The given data is split into training and testing with a ratio of 80/20. Accuracy Score is reported from test data.