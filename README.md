
# Neural Network with numpy

Implementation of a feedforward neural network for classification tasks using gradient descent optimization

- By default, the network is designed for binary classification, with the output layer using sigmoid for the activation function and binary cross-entropy for loss function.

- By setting binary_classification to false in the initialization, it will automatically use softmax for the output layer

# How to run the project

First you need to create a Python virtual environment

    $ python3 -m venv venv

Activate the virtual environment

    Linux/MacOS $ source venv/bin/activate
    
    Windows $ venv\Scripts\activate

Install the requirements

    $ pip install -r requirements.txt


To train and test the model

    $ python3 training.py

Deactivate the virtual environment after you're done working on the project

    $ deactivate
