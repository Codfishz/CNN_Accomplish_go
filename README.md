# CNN_Accomplish_go
Group project for 02-601
Project overview:
In this project, we aim to employ CNN technology for the detection of brain tumors through the analysis of MRI images. Our approach will begin by delving into the fundamental principles of the CNN algorithm and implementing key functions, including convolution, pooling, activation, and backpropagation, starting from the ground up.
Our initial step involves constructing a CNN model, which we will then train using the MNIST dataset and evaluate the model's ability to recognize and distinguish handwritten digits. After confirming the correctness and efficacy of the model using the MNIST dataset, we will use a dataset dedicated to MRI images of brain tumors to train our model specifically for brain tumor detection, putting its capabilities to the test in a real-world scenario.



Usage:
Make sure the path is correct and use go build to compile. The program will go to the specified path to read the data set file and use the data set to train and evaluate the CNN model.
The program will sequentially perform training and evaluation on the MNIST data set; training and evaluation on the Brain tumor data set. The training results will be reflected in the command line.
Two parts are implemented using Python: one is the normalization of the brain tumor data set(load_image.ipynb), and the other is the visualization of changes in accuracy and loss during the training process(visualization.ipynb).

