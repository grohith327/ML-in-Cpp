# Linear Regression
A C++ program that performs a simple linear regression and returns the mse error </br>
### Ensure the path for mlpack has been set: </br>
By default mlpack is installed to `/usr/local/`, if you have installed somewhere specify that path. 
```
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
```
### Compile Program with the following command: </br>
```
clang++ -L/usr/local/lib -l mlpack -l armadillo -l boost_serialization -l boost_program_options -I/usr/local/include Linear_Regression.cpp
```
### Run: </br>
The program takes three command line arguments </br>
* Mandatory:</br>
    * `features.csv` - input features
    * `responses.csv` - predictors
* Optional:
    * `lambda` - regularisation parameter for ridge regression

```
./a.out <features.csv> <responses.csv> 0.1
```

#### Note: </br>
The given data is split into training and testing with a ratio of 80/20. MSE is reported from test data.