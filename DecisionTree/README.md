# Decision Tree
A C++ program that train a Decision Tree classifier and returns the accuracy of the model </br>
### Ensure the path for mlpack has been set: </br>
By default mlpack is installed to `/usr/local/`, if you have installed somewhere specify that path. 
```
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
```
### Compile Program with the following command: </br>
```
clang++ -L/usr/local/lib -l mlpack -l armadillo -l boost_serialization -l boost_program_options -I/usr/local/include DecisionTree.cpp
```
### Run: </br>
The program takes one command line argument </br>
* `data.csv` - data with input features and labels

```
./a.out <data.csv>
```

#### Note: </br>
* The given data is split into training and testing with a ratio of 80/20. Accuracy is reported from test data.
* The last attribute of the dataset represents the label.