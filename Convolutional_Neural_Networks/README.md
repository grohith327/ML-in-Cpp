# Convolutional Neural Network
A C++ program that performs classification using a Convolutional Neural Network and returns the accuracy of the model</br>
### Ensure the path for mlpack has been set: </br>
By default mlpack is installed to `/usr/local/`, if you have installed somewhere specify that path. 
```
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
```
### Compile Program with the following command: </br>
```
clang++ -L/usr/local/lib -l mlpack -l armadillo -l boost_serialization -l boost_program_options -I/usr/local/include CNN.cpp
```
### Run: </br>
The program takes four command line arguments </br>
* `data.csv` - input data with last column as label
* `epochs` - number of epochs to train
* `image width` - width of the image
* `image height` - height of the image

```
./a.out <data.csv> <epochs> <img_width> <img_height>
```

#### Note: </br>
* The given data is split into training and testing with a ratio of 80/20. Accuracy Score is reported from test data.
* Model is trained using a standard Gradient Descent optimizer and a log likelihood loss function
* The pixel values of the image is loaded from a csv file
#### TODO
Add support for loading images directly instead of csv file. 